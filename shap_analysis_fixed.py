#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v1 - SHAP 特征重要性分析与可视化 (修复版)
修复:
  1. 自动设置 _text 列（transform 必需）
  2. title 改用 plt.title()（shap.summary_plot 的 title 参数无效）
  3. GradientExplainer -> DeepExplainer（Transformer 更稳定）
  4. 输出三分类全部 SHAP 图 + CSV
"""
import os
import torch
import shap
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_feature_names():
    """32维特征名称，与 SOCFeatureEncoder 输出顺序一致"""
    time_feats = ['time_hour_norm', 'time_sin', 'time_cos', 'time_is_weekend', 'time_is_night']
    cat_feats = ['pipeline_idx', 'username_idx', 'src_host_idx', 'src_ip_idx']
    ip_feats = ['ip_is_private', 'ip_subnet_hash', 'ip_is_loopback']
    msg_feats = ['msg_length', 'msg_is_missing', 'kw_malicious_count', 'kw_benign_count']
    tfidf_feats = [f'tfidf_pca_{i}' for i in range(16)]
    return time_feats + cat_feats + ip_feats + msg_feats + tfidf_feats


def prepare_text_column(df):
    """
    与 SOCFeatureEncoder.fit_transform() 中的 _text 设置逻辑完全一致。
    transform() 依赖 _text 列计算 msg_feats 和 tfidf_feats。
    """
    if 'message_sanitized' in df.columns:
        df['_text'] = df['message_sanitized'].fillna('').astype(str)
    elif 'product_name' in df.columns and 'vendor_name' in df.columns:
        df['_text'] = (df['product_name'].fillna('') + ' ' + df['vendor_name'].fillna('')).str.strip()
    elif 'product_name' in df.columns:
        df['_text'] = df['product_name'].fillna('').astype(str)
    else:
        df['_text'] = ''
    return df


def main():
    p = argparse.ArgumentParser(description='Trans-CL v1 SHAP Analysis')
    p.add_argument('--data-path', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--encoder-path', required=True)
    p.add_argument('--num-background', type=int, default=200)
    p.add_argument('--num-test', type=int, default=100)
    p.add_argument('--output-dir', default='shap_output')
    p.add_argument('--class-index', type=int, default=None,
                   help='指定单类别 (0=benign,1=suspicious,2=malicious)。默认分析全部3类')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    print(f'[1/5] Loading data: {args.data_path}')
    if args.data_path.endswith(('.parquet', '.pq')):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path, low_memory=False)

    # 关键修复: 设置 _text 列，否则 TF-IDF + 消息特征全为0
    df = prepare_text_column(df)
    print(f'       Total: {len(df):,} samples | _text column ready')

    # 2. 加载 Encoder
    print('[2/5] Loading encoder...')
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'train_transcl_v1'
            return super().find_class(module, name)

    with open(args.encoder_path, 'rb') as f:
        encoder = CustomUnpickler(f).load()

    # 3. 加载模型
    print('[3/5] Loading model...')
    from train_transcl_v1 import TransCLContrastiveModel
    model = TransCLContrastiveModel(device=device, num_classes=3)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print('       Model loaded & eval mode')

    # 4. 特征转换 & 采样
    print('[4/5] Encoding features...')
    total_needed = min(len(df), args.num_background + args.num_test)
    df_sample = df.sample(n=total_needed, random_state=42)
    feats = encoder.transform(df_sample)
    X_tensor = torch.from_numpy(feats).float().to(device)

    X_background = X_tensor[:args.num_background]
    X_test = X_tensor[args.num_background: args.num_background + args.num_test]
    feature_names = get_feature_names()
    print(f'       Background: {X_background.shape[0]} | Test: {X_test.shape[0]}')

    # 5. SHAP 计算
    print('[5/5] Computing SHAP values (may take a few minutes)...')
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_test)

    X_test_np = X_test.cpu().numpy()
    class_names = ['Benign', 'Suspicious', 'Malicious']

    # 6. 可视化 & 保存
    target_classes = [args.class_index] if args.class_index is not None else [0, 1, 2]

    for cls_idx in target_classes:
        sv = shap_values[cls_idx]
        if isinstance(sv, torch.Tensor):
            sv = sv.cpu().detach().numpy()

        # --- Beeswarm Summary Plot ---
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_test_np, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary for {class_names[cls_idx]} Class (n={X_test_np.shape[0]})', fontsize=14)
        plt.tight_layout()
        path = os.path.join(args.output_dir, f'shap_summary_{class_names[cls_idx].lower()}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'       Saved: {path}')

        # --- Bar Plot (mean |SHAP|) ---
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_test_np, feature_names=feature_names, show=False, plot_type='bar')
        plt.title(f'Mean |SHAP| for {class_names[cls_idx]} Class', fontsize=14)
        plt.tight_layout()
        path = os.path.join(args.output_dir, f'shap_bar_{class_names[cls_idx].lower()}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'       Saved: {path}')

        # --- CSV Export ---
        mean_abs_shap = np.abs(sv).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap,
            'sum_shap': sv.sum(axis=0),
            'pos_shap': np.where(sv > 0, sv, 0).sum(axis=0),
            'neg_shap': np.where(sv < 0, sv, 0).sum(axis=0),
        }).sort_values('mean_abs_shap', ascending=False)
        csv_path = os.path.join(args.output_dir, f'shap_importance_{class_names[cls_idx].lower()}.csv')
        importance_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f'       Saved: {csv_path}')

    print(f'\n[INFO] All outputs in: {args.output_dir}/')


if __name__ == '__main__':
    main()
