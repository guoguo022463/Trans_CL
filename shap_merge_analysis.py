#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP 特征重要性 + 冗余性分析（用于判断哪些字段可合并）

说明：
  - TransCL Transformer 模型用 shap.DeepExplainer 会触发“可加性校验失败”，
    因此这里用 shap.TreeExplainer + 随机森林代理模型对同一套 34 维特征做
    可靠的表格特征重要性分析。
  - 同时输出特征两两相关性，辅助判断冗余字段。
"""
import os
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import shap

from train_transcl_v1 import SOCFeatureEncoder

warnings.filterwarnings('ignore')


FEATURE_NAMES = [
    'time_hour_norm', 'time_sin', 'time_cos', 'time_is_weekend', 'time_is_night',
    'pipeline_idx', 'username_idx', 'src_host_idx', 'src_ip_idx',
    'ip_is_private', 'ip_subnet_hash', 'ip_is_loopback',
    'msg_length', 'msg_is_missing', 'kw_malicious_count', 'kw_benign_count',
    'dst_host_missing', 'username_missing',
    'tfidf_pca_0', 'tfidf_pca_1', 'tfidf_pca_2', 'tfidf_pca_3',
    'tfidf_pca_4', 'tfidf_pca_5', 'tfidf_pca_6', 'tfidf_pca_7',
    'tfidf_pca_8', 'tfidf_pca_9', 'tfidf_pca_10', 'tfidf_pca_11',
    'tfidf_pca_12', 'tfidf_pca_13', 'tfidf_pca_14', 'tfidf_pca_15',
]

GROUPS = {
    'time': [0, 1, 2, 3, 4],
    'cat_id': [5, 6, 7, 8],
    'ip': [9, 10, 11],
    'msg': [12, 13, 14, 15],
    'missing': [16, 17],
    'tfidf_pca': list(range(18, 34)),
}

CLASS_NAMES = ['Benign', 'Suspicious', 'Malicious']


def read_table(path):
    if str(path).endswith(('.parquet', '.pq')):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for col in df.columns:
        arr = df[col].array
        if 'Arrow' in type(arr).__name__ or hasattr(arr, '_pa_array'):
            df[col] = pd.Series(arr.to_numpy(), dtype='object')
    return df


def attach_label(df):
    label_col = 'label_binary' if 'label_binary' in df.columns else 'label'
    assert label_col in df.columns, f'No label column, got {df.columns.tolist()}'
    label_map = {'benign': 0, 'suspicious': 1, 'malicious': 2}
    if df[label_col].dtype == object or str(df[label_col].dtype).startswith('str'):
        for v in df[label_col].unique():
            if v not in label_map:
                label_map[v] = 0
        df['label'] = df[label_col].map(label_map).astype(np.int64)
    else:
        df['label'] = df[label_col].astype(np.int64)
    return df


def balanced_sample(df, per_class):
    parts = []
    for c in [0, 1, 2]:
        sub = df[df['label'] == c]
        if len(sub) > per_class:
            sub = sub.sample(n=per_class, random_state=42)
        parts.append(sub)
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--per-class', type=int, default=10000)
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--output-dir', default='shap_merge_output')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print('[1/4] Loading & labeling data...')
    df = attach_label(read_table(args.data_path))
    df = balanced_sample(df, args.per_class)
    print(f'  balanced sample: {len(df):,} rows | {df.label.value_counts().to_dict()}')

    print('[2/4] Encoding 34 features...')
    encoder = SOCFeatureEncoder()
    X = encoder.fit_transform(df)
    y = df['label'].values.astype(np.int64)
    print(f'  X={X.shape}')

    print('[3/4] Training RandomForest surrogate...')
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight='balanced',
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X, y)

    print('[4/4] Computing SHAP + correlations...')
    # 用较小测试集算 SHAP，避免内存爆炸
    n_test = min(2000, len(X))
    idx = np.random.RandomState(args.seed).choice(len(X), size=n_test, replace=False)
    X_test = X[idx]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    # 兼容两种返回格式：
    #  - list[class] -> (n_test, 34)
    #  - ndarray (n_test, 34, n_classes)
    if isinstance(shap_values, list):
        shap_per_class = [np.asarray(shap_values[c]) for c in range(3)]
    else:
        shap_per_class = [np.asarray(shap_values)[..., c] for c in range(3)]

    # 每个类别 mean |SHAP|
    per_class_importance = {}
    for c in range(3):
        sv = shap_per_class[c]
        imp = np.abs(sv).mean(axis=0)
        per_class_importance[CLASS_NAMES[c]] = {
            FEATURE_NAMES[i]: float(imp[i]) for i in range(len(FEATURE_NAMES))
        }

    # 总体 mean |SHAP|（对类别求平均）
    overall = np.mean([np.abs(shap_per_class[c]).mean(axis=0) for c in range(3)], axis=0)
    overall_imp = {FEATURE_NAMES[i]: float(overall[i]) for i in range(len(FEATURE_NAMES))}

    # 相关性矩阵
    corr = np.corrcoef(X, rowvar=False)
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            r = abs(corr[i, j])
            if r >= 0.8:
                high_corr_pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], float(r)))
    high_corr_pairs.sort(key=lambda x: -x[2])

    # 汇总导出
    ranked = sorted(overall_imp.items(), key=lambda kv: -kv[1])
    rows = []
    for name, imp in ranked:
        rows.append({
            'feature': name,
            'mean_abs_shap': round(imp, 6),
            'benign': round(per_class_importance['Benign'][name], 6),
            'suspicious': round(per_class_importance['Suspicious'][name], 6),
            'malicious': round(per_class_importance['Malicious'][name], 6),
            'group': next(g for g, idxs in GROUPS.items() if FEATURE_NAMES.index(name) in idxs),
        })
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(os.path.join(args.output_dir, 'shap_importance.csv'), index=False)

    corr_df = pd.DataFrame(corr, index=FEATURE_NAMES, columns=FEATURE_NAMES)
    corr_df.to_csv(os.path.join(args.output_dir, 'feature_correlation.csv'))

    with open(os.path.join(args.output_dir, 'high_corr_pairs.json'), 'w', encoding='utf-8') as f:
        json.dump(high_corr_pairs, f, ensure_ascii=False, indent=2)

    # 打印
    print('\n===== 特征重要性排名（mean |SHAP|，跨3类平均）=====')
    print(f'{"feature":24s} {"mean|SHAP|":>12s} {"benign":>10s} {"susp":>10s} {"mal":>10s}  group')
    for r in rows:
        print(f'{r["feature"]:24s} {r["mean_abs_shap"]:>12.4f} {r["benign"]:>10.4f} '
              f'{r["suspicious"]:>10.4f} {r["malicious"]:>10.4f}  {r["group"]}')

    print(f'\n===== |相关系数| >= 0.8 的特征对（共 {len(high_corr_pairs)} 对）=====')
    for a, b, r in high_corr_pairs[:40]:
        print(f'  {a:22s} <-> {b:22s}  r={r:.4f}')

    print(f'\n[INFO] 结果已保存到 {args.output_dir}/')


if __name__ == '__main__':
    main()
