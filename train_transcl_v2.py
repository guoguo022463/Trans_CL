#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL SOC 分类训练脚本 v3 (Two-Stage)
=============================================
Phase 1: 二分类预训练 (benign vs attack) — 全层训练
Phase 2: 三分类微调 (benign/suspicious/malicious) — 冻结Encoder，训练分类头

特征工程 (32维):
  改进1 [时间丰富化]: hour, hour_sin, hour_cos, is_weekend, is_night  (5维)
  原始+改进2 [IP信息]: pipeline_id, username_id, src_host_id, src_ip_id,
                     is_private_ip, ip_subnet_id, is_loopback  (7维)
  原始+改进4-C [关键词]: msg_len, msg_missing, kw_malicious, kw_benign  (4维)
  改进4-A [TF-IDF PCA]: tfidf_pca_0..15  (16维)
  ----------------------------------------
  总计: 32维
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import json
import csv
import time
import warnings
import re
import ipaddress
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

from transcl_classifier import TransCL_SOCClassifier, FocalLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Run: pip install wandb")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOT = True
except ImportError:
    MATPLOT = False


# ============================================================
# 1. 数据加载
# ============================================================

def load_data(path):
    print('Loading data from %s...' % path)
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    print('  -> %d rows, columns: %s' % (len(df), list(df.columns)))

    # 兼容 competition 数据：label_binary -> label
    if 'label' not in df.columns and 'label_binary' in df.columns:
        label_map = {'benign': 0, 'suspicious': 1, 'malicious': 2}
        df['label'] = df['label_binary'].map(label_map)

    if 'label' not in df.columns:
        print('  [WARNING] No "label" or "label_binary" column found. This file will be treated as unlabeled.')
        return df
    df['label'] = df['label'].astype(int)

    counts = df['label'].value_counts().sort_index()
    label_names = {0: 'Benign', 1: 'Suspicious', 2: 'Malicious'}
    for idx in [0, 1, 2]:
        if idx in counts:
            print('  %s: %d' % (label_names.get(idx, 'Class%d' % idx), counts[idx]))
    return df


# ============================================================
# 2. 增强版特征编码器 (32维)
# ============================================================

class SOCFeatureEncoder:
    """
    SOC日志特征编码器 v3 (32维)

    特征索引映射:
      0  hour              : timestamp -> hour/24
      1  hour_sin          : sin(2*pi*hour/24)
      2  hour_cos          : cos(2*pi*hour/24)
      3  is_weekend        : weekday >= 5 ? 1 : 0
      4  is_night          : hour < 6 or hour > 22 ? 1 : 0
      5  pipeline_id       : pipeline 字符串映射 (分类列)
      6  username_id       : username 字符串映射 (分类列)
      7  src_host_id       : src_host 字符串映射 (分类列)
      8  src_ip_id         : src_ip 字符串映射 (分类列)
      9  is_private_ip     : IP是否私有地址
      10 ip_subnet_id      : /24子网hash值 (归一化到0-1)
      11 is_loopback       : IP是否回环地址
      12 msg_len           : message长度 / 1000 (cap=5.0)
      13 msg_missing       : message是否缺失
      14 kw_malicious      : 恶意关键词命中数
      15 kw_benign         : 良性关键词命中数
      16-31 tfidf_pca_0..15: TF-IDF PCA降维 (16维)
    """
    NUM_FEATURES = 32
    NUM_COLUMNS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31]  # 28个数值列
    CAT_COLUMNS = [5, 6, 7, 8]  # 4个分类列

    def __init__(self):
        # 字符串映射表
        self.pipeline_map = {'__MISSING__': 0, '__UNK__': 1}
        self.username_map = {'__MISSING__': 0, '__UNK__': 1}
        self.src_host_map = {'__MISSING__': 0, '__UNK__': 1}
        self.src_ip_map = {'__MISSING__': 0, '__UNK__': 1}

        # 硬编码关键词（改进4-C前向兼容，fit时自动扩展）
        self.malicious_kws_hard = [
            'unauthorized', 'malicious', 'escalation', 'injection',
            'brute force', 'exfiltration', 'attack', 'blocked',
            'denied', 'failed', 'violation', 'breach', 'compromised',
            'suspicious', 'alert', 'critical', 'error', 'anomaly'
        ]
        self.benign_kws_hard = [
            'login successful', 'normal', 'logged out', 'restarted',
            'success', 'allowed', 'completed', 'authenticated',
            'approved', 'established'
        ]
        # 自动扩展的关键词表（fit后填充）
        self.malicious_kws_auto = []
        self.benign_kws_auto = []

        # TF-IDF + PCA（fit后填充）
        self.tfidf_vectorizer = None
        self.tfidf_pca = None
        self.tfidf_dim = 16

    # ---------- fit 相关方法 ----------

    def fit(self, df):
        """学习所有映射和统计量"""
        print('Fitting encoder...')

        # 1. 分类列映射
        for col, mapping in [('username', self.username_map),
                             ('src_host', self.src_host_map),
                             ('src_ip', self.src_ip_map)]:
            if col in df.columns:
                uniques = df[col].fillna('').replace('', '__MISSING__').unique()
                mapping.clear()
                mapping['__MISSING__'] = 0
                mapping['__UNK__'] = 1
                for i, v in enumerate(uniques):
                    if v not in ('__MISSING__', '__UNK__'):
                        mapping[v] = i + 2
        if 'pipeline' in df.columns:
            pipelines = df['pipeline'].fillna('').replace('', '__MISSING__').unique()
            self.pipeline_map.clear()
            self.pipeline_map['__MISSING__'] = 0
            self.pipeline_map['__UNK__'] = 1
            for i, v in enumerate(pipelines):
                if v not in ('__MISSING__', '__UNK__'):
                    self.pipeline_map[v] = i + 2

        # 2. 自动扩展关键词表 (改进4-C)
        self._fit_keywords(df)

        # 3. TF-IDF + PCA (改进4-A)
        self._fit_tfidf_pca(df)

        print('Encoder fitted:')
        print('  pipeline: %d categories' % len(self.pipeline_map))
        print('  username: %d categories' % len(self.username_map))
        print('  src_host: %d categories' % len(self.src_host_map))
        print('  src_ip: %d categories' % len(self.src_ip_map))
        print('  malicious keywords (auto): %d (+%d hard-coded)' % (
            len(self.malicious_kws_auto), len(self.malicious_kws_hard)))
        print('  benign keywords (auto): %d (+%d hard-coded)' % (
            len(self.benign_kws_auto), len(self.benign_kws_hard)))
        print('  TF-IDF PCA dim: %d' % self.tfidf_dim)

    def _fit_keywords(self, df):
        """改进4-C: 从训练集自动挖掘高区分度关键词"""
        if 'label' not in df.columns or 'message_sanitized' not in df.columns:
            return

        # 分词函数
        def tokenize(text):
            text = str(text).lower()
            # 保留字母数字，按非字母数字分割
            words = re.findall(r'[a-z]+', text)
            # 过滤停用词和过短词
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                         'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below',
                         'between', 'under', 'and', 'but', 'or', 'yet', 'so', 'if',
                         'because', 'although', 'though', 'while', 'where', 'when',
                         'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                         'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
                         'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
                         'its', 'they', 'them', 'their', 'user', 'host', 'ip',
                         'port', 'id', 'name', 'time', 'date', 'log', 'event'}
            return [w for w in words if len(w) > 3 and w not in stopwords]

        # 统计各类词频
        benign_words = Counter()
        mal_words = Counter()

        benign_df = df[df['label'] == 0]
        mal_df = df[df['label'] == 2]

        for msg in benign_df['message_sanitized'].fillna('').astype(str):
            benign_words.update(tokenize(msg))
        for msg in mal_df['message_sanitized'].fillna('').astype(str):
            mal_words.update(tokenize(msg))

        # 提取恶意类高区分度词: 在恶意类高频，但在良性类低频
        self.malicious_kws_auto = []
        for word, count in mal_words.most_common(300):
            ben_count = benign_words.get(word, 0)
            # 条件: 恶意词频 > 10 且 (良性词频 / 恶意词频) < 0.3
            if count >= 10 and ben_count < count * 0.3:
                self.malicious_kws_auto.append(word)
            if len(self.malicious_kws_auto) >= 50:
                break

        # 提取良性类高区分度词
        self.benign_kws_auto = []
        for word, count in benign_words.most_common(300):
            mal_count = mal_words.get(word, 0)
            if count >= 100 and mal_count < count * 0.05:
                self.benign_kws_auto.append(word)
            if len(self.benign_kws_auto) >= 50:
                break

    def _fit_tfidf_pca(self, df):
        """改进4-A: 训练 TF-IDF + PCA"""
        if 'message_sanitized' not in df.columns:
            return

        messages = df['message_sanitized'].fillna('').astype(str).tolist()
        # 简单预处理：小写，保留字母数字和空格
        messages = [re.sub(r'[^a-z0-9\s]', ' ', m.lower()) for m in messages]

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=5,           # 至少在5个文档中出现
            max_df=0.8,         # 最多在80%的文档中出现（过滤停用词）
            ngram_range=(1, 2), # 一元和二元
            stop_words='english'
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(messages)

        # PCA 降维
        self.tfidf_pca = PCA(n_components=self.tfidf_dim)
        self.tfidf_pca.fit(tfidf_matrix.toarray())

        explained = sum(self.tfidf_pca.explained_variance_ratio_)
        print('  TF-IDF vocab: %d -> PCA(%d) explained variance: %.2f%%' % (
            len(self.tfidf_vectorizer.vocabulary_), self.tfidf_dim, explained * 100))

    # ---------- transform 相关方法 ----------

    def _safe_get(self, row, col):
        val = row.get(col, '')
        if pd.isna(val):
            return '__MISSING__'
        val = str(val).strip()
        return val if val != '' else '__MISSING__'

    def _extract_time_features(self, ts_val):
        """改进1: 时间特征丰富化 (5维)"""
        try:
            if isinstance(ts_val, str):
                ts_val = float(ts_val)
            dt = datetime.fromtimestamp(ts_val)
            hour = dt.hour
            return [
                hour / 24.0,                                    # 0: hour
                np.sin(2 * np.pi * hour / 24),                  # 1: hour_sin
                np.cos(2 * np.pi * hour / 24),                  # 2: hour_cos
                1.0 if dt.weekday() >= 5 else 0.0,              # 3: is_weekend
                1.0 if (hour < 6 or hour > 22) else 0.0,        # 4: is_night
            ]
        except Exception:
            return [0.0, 0.0, 1.0, 0.0, 0.0]

    def _extract_ip_features(self, src_ip):
        """改进2: IP信息提取 (4维: src_ip_id外，新增3维)"""
        is_private = 0.0
        is_loopback = 0.0
        subnet_hash = 0.0
        try:
            ip = ipaddress.ip_address(str(src_ip))
            is_private = 1.0 if ip.is_private else 0.0
            is_loopback = 1.0 if ip.is_loopback else 0.0
            # /24 子网 hash 归一化
            network = ipaddress.ip_network(str(ip) + '/24', strict=False)
            subnet_hash = hash(str(network.network_address)) % 1000 / 1000.0
        except Exception:
            pass
        return is_private, subnet_hash, is_loopback

    def _extract_msg_features(self, row):
        """msg相关特征 (4维) + TF-IDF (16维)"""
        msg = str(row.get('message_sanitized', '')) if pd.notna(row.get('message_sanitized')) else ''
        msg_len = min(len(msg) / 1000.0, 5.0)
        msg_missing = 1.0 if (pd.isna(row.get('message_sanitized')) or str(row.get('message_sanitized', '')).strip() == '') else 0.0
        msg_lower = msg.lower()

        # 关键词计数: 硬编码 + 自动扩展
        all_mal_kws = self.malicious_kws_hard + self.malicious_kws_auto
        all_ben_kws = self.benign_kws_hard + self.benign_kws_auto
        kw_malicious = sum(1 for kw in all_mal_kws if kw in msg_lower)
        kw_benign = sum(1 for kw in all_ben_kws if kw in msg_lower)

        return msg_len, msg_missing, kw_malicious, kw_benign

    def _extract_tfidf_features(self, msg):
        """改进4-A: TF-IDF PCA特征 (16维)"""
        if self.tfidf_vectorizer is None or self.tfidf_pca is None:
            return [0.0] * self.tfidf_dim

        msg_clean = re.sub(r'[^a-z0-9\s]', ' ', str(msg).lower())
        tfidf_vec = self.tfidf_vectorizer.transform([msg_clean])
        pca_vec = self.tfidf_pca.transform(tfidf_vec.toarray())[0]
        return pca_vec.tolist()

    def transform_single(self, row):
        """单条日志 -> 32维特征向量"""
        # 时间特征 (5维)
        time_feats = self._extract_time_features(row.get('timestamp', 0))

        # 分类列映射
        pipeline = self._safe_get(row, 'pipeline')
        pipeline_id = self.pipeline_map.get(pipeline, self.pipeline_map.get('__UNK__', 1))
        username = self._safe_get(row, 'username')
        username_id = self.username_map.get(username, self.username_map.get('__UNK__', 1))
        src_host = self._safe_get(row, 'src_host')
        src_host_id = self.src_host_map.get(src_host, self.src_host_map.get('__UNK__', 1))
        src_ip = self._safe_get(row, 'src_ip')
        src_ip_id = self.src_ip_map.get(src_ip, self.src_ip_map.get('__UNK__', 1))

        # IP扩展特征 (3维)
        is_private, subnet_hash, is_loopback = self._extract_ip_features(src_ip)

        # msg特征 (4维)
        msg_len, msg_missing, kw_malicious, kw_benign = self._extract_msg_features(row)

        # TF-IDF PCA (16维)
        msg = str(row.get('message_sanitized', '')) if pd.notna(row.get('message_sanitized')) else ''
        tfidf_feats = self._extract_tfidf_features(msg)

        # 组合 (32维)
        features = [
            time_feats[0],   # 0  hour
            time_feats[1],   # 1  hour_sin
            time_feats[2],   # 2  hour_cos
            time_feats[3],   # 3  is_weekend
            time_feats[4],   # 4  is_night
            float(pipeline_id),  # 5  pipeline_id (分类)
            float(username_id),  # 6  username_id (分类)
            float(src_host_id),  # 7  src_host_id (分类)
            float(src_ip_id),    # 8  src_ip_id (分类)
            is_private,      # 9
            subnet_hash,     # 10
            is_loopback,     # 11
            msg_len,         # 12
            msg_missing,     # 13
            float(kw_malicious), # 14
            float(kw_benign),    # 15
        ] + tfidf_feats    # 16-31

        return np.array(features, dtype=np.float32)

    def transform(self, df):
        features = np.stack([self.transform_single(row) for _, row in df.iterrows()])
        return features

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


# ============================================================
# 3. 评估指标（完整二级+三级）
# ============================================================

def evaluate_complete_metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    acc = float((preds == labels).mean())
    per_class = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        eps = 1e-8
        tpr = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        tnr = tn / (tn + fp + eps)
        fnr = fn / (fn + tp + eps)
        pre = tp / (tp + fp + eps)
        sen = tpr
        f1 = 2 * pre * sen / (pre + sen + eps)
        per_class['class_%d' % c] = {
            'TPR': round(float(tpr), 4), 'FPR': round(float(fpr), 4),
            'TNR': round(float(tnr), 4), 'FNR': round(float(fnr), 4),
            'PRE': round(float(pre), 4), 'SEN': round(float(sen), 4),
            'F1': round(float(f1), 4),
        }
    macro_tpr = float(np.mean([per_class['class_%d' % c]['TPR'] for c in range(3)]))
    macro_fpr = float(np.mean([per_class['class_%d' % c]['FPR'] for c in range(3)]))
    macro_tnr = float(np.mean([per_class['class_%d' % c]['TNR'] for c in range(3)]))
    macro_fnr = float(np.mean([per_class['class_%d' % c]['FNR'] for c in range(3)]))
    macro_pre = float(np.mean([per_class['class_%d' % c]['PRE'] for c in range(3)]))
    macro_sen = float(np.mean([per_class['class_%d' % c]['SEN'] for c in range(3)]))
    macro_f1 = float(np.mean([per_class['class_%d' % c]['F1'] for c in range(3)]))
    class_counts = np.bincount(labels, minlength=3).astype(float)
    total_c = class_counts.sum()
    weights_c = class_counts / total_c if total_c > 0 else np.ones(3) / 3
    w_tpr = float(np.sum([weights_c[c] * per_class['class_%d' % c]['TPR'] for c in range(3)]))
    w_fpr = float(np.sum([weights_c[c] * per_class['class_%d' % c]['FPR'] for c in range(3)]))
    w_tnr = float(np.sum([weights_c[c] * per_class['class_%d' % c]['TNR'] for c in range(3)]))
    w_fnr = float(np.sum([weights_c[c] * per_class['class_%d' % c]['FNR'] for c in range(3)]))
    w_pre = float(np.sum([weights_c[c] * per_class['class_%d' % c]['PRE'] for c in range(3)]))
    w_sen = float(np.sum([weights_c[c] * per_class['class_%d' % c]['SEN'] for c in range(3)]))
    w_f1 = float(np.sum([weights_c[c] * per_class['class_%d' % c]['F1'] for c in range(3)]))
    try:
        binary_labels = (labels > 0).astype(int)
        binary_probs = 1 - probs[:, 0]
        auc = float(roc_auc_score(binary_labels, binary_probs))
    except Exception:
        auc = None
    return {
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'ACC': round(acc, 4),
        'Macro': {
            'TPR': round(macro_tpr, 4), 'FPR': round(macro_fpr, 4),
            'TNR': round(macro_tnr, 4), 'FNR': round(macro_fnr, 4),
            'PRE': round(macro_pre, 4), 'SEN': round(macro_sen, 4),
            'F1': round(macro_f1, 4),
        },
        'Weighted': {
            'TPR': round(w_tpr, 4), 'FPR': round(w_fpr, 4),
            'TNR': round(w_tnr, 4), 'FNR': round(w_fnr, 4),
            'PRE': round(w_pre, 4), 'SEN': round(w_sen, 4),
            'F1': round(w_f1, 4),
        },
        'AUC': round(auc, 4) if auc is not None else None,
    }


def print_metrics(result, title='Evaluation'):
    cm = result['confusion_matrix']
    names = ['Benign', 'Suspicious', 'Malicious']
    print()
    print('=' * 85)
    print('  %s' % title)
    print('=' * 85)
    print('  Confusion Matrix:')
    print('  %-12s %8s %8s %8s' % ('', 'Pred_Ben', 'Pred_Sus', 'Pred_Mal'))
    for i in range(3):
        print('  %-12s %8d %8d %8d' % ('True_' + names[i], cm[i][0], cm[i][1], cm[i][2]))
    print()
    print('  [二级指标] TPR / FPR / TNR / FNR')
    print('  %-12s %8s %8s %8s %8s' % ('Class', 'TPR', 'FPR', 'TNR', 'FNR'))
    print('  ' + '-' * 50)
    for c in range(3):
        m = result['per_class']['class_%d' % c]
        print('  %-12s %8.4f %8.4f %8.4f %8.4f' % (names[c], m['TPR'], m['FPR'], m['TNR'], m['FNR']))
    print('  ' + '-' * 50)
    m2 = result['Macro']
    w2 = result['Weighted']
    print('  %-12s %8.4f %8.4f %8.4f %8.4f' % ('Macro', m2['TPR'], m2['FPR'], m2['TNR'], m2['FNR']))
    print('  %-12s %8.4f %8.4f %8.4f %8.4f' % ('Weighted', w2['TPR'], w2['FPR'], w2['TNR'], w2['FNR']))
    print()
    print('  [三级指标] PRE / SEN / F1')
    print('  %-12s %8s %8s %8s' % ('Class', 'PRE', 'SEN', 'F1'))
    print('  ' + '-' * 40)
    for c in range(3):
        m = result['per_class']['class_%d' % c]
        print('  %-12s %8.4f %8.4f %8.4f' % (names[c], m['PRE'], m['SEN'], m['F1']))
    print('  ' + '-' * 40)
    print('  %-12s %8.4f %8.4f %8.4f' % ('Macro', m2['PRE'], m2['SEN'], m2['F1']))
    print('  %-12s %8.4f %8.4f %8.4f' % ('Weighted', w2['PRE'], w2['SEN'], w2['F1']))
    print()
    print('  ACC  = %.4f' % result['ACC'])
    if result['AUC'] is not None:
        print('  AUC  = %.4f' % result['AUC'])
    print('=' * 85)


# ============================================================
# 4. W&B 日志辅助函数
# ============================================================

def log_confusion_matrix_to_wandb(wandb_run, cm, class_names, step=None):
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    table_data = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            table_data.append([true_name, pred_name, cm[i][j]])
    table = wandb.Table(columns=['true_label', 'pred_label', 'count'], data=table_data)
    wandb_run.log({'evaluation/confusion_matrix': table}, step=step)


def log_per_class_metrics_to_wandb(wandb_run, result, prefix='validation', step=None):
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    names = ['benign', 'suspicious', 'malicious']
    for c in range(3):
        m = result['per_class']['class_%d' % c]
        for metric_name in ['TPR', 'FPR', 'TNR', 'FNR', 'PRE', 'SEN', 'F1']:
            wandb_run.log({
                '%s/%s_%s' % (prefix, names[c], metric_name): m[metric_name]
            }, step=step)


def log_artifact_to_wandb(wandb_run, filepath, artifact_name, artifact_type='model', metadata=None):
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {}
    )
    artifact.add_file(filepath)
    wandb_run.log_artifact(artifact)
    print('  [W&B] Artifact uploaded: %s' % artifact_name)


# ============================================================
# 5. 类别权重计算
# ============================================================

def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * 3.0)
    weights = np.minimum(weights, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 6. 实验日志（本地）
# ============================================================

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        self.start_time = time.time()

    def log_hyperparams(self, hparams):
        path = os.path.join(self.log_dir, 'hyperparams.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(hparams, f, indent=2, ensure_ascii=False)

    def log_epoch(self, epoch, train_loss, train_acc, eval_result=None, lr=None, epoch_time=None):
        record = {'epoch': epoch, 'train_loss': round(train_loss, 4), 'train_acc': round(train_acc, 4), 'lr': lr}
        if epoch_time is not None:
            record['epoch_time_sec'] = round(epoch_time, 2)
        if eval_result is not None:
            record['val_acc'] = eval_result['ACC']
            record['val_pre'] = eval_result['Weighted']['PRE']
            record['val_sen'] = eval_result['Weighted']['SEN']
            record['val_spe'] = eval_result['Weighted']['TNR']
            record['val_f1'] = eval_result['Weighted']['F1']
            if eval_result['AUC'] is not None:
                record['val_auc'] = eval_result['AUC']
        self.metrics.append(record)
        if self.metrics:
            path = os.path.join(self.log_dir, 'metrics.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.metrics)

    def log_summary(self, best_f1, best_epoch):
        total_time = time.time() - self.start_time
        summary = {'best_weighted_f1': best_f1, 'best_epoch': best_epoch, 'total_time_sec': round(total_time, 2)}
        if self.metrics:
            summary['final_train_acc'] = self.metrics[-1]['train_acc']
        path = os.path.join(self.log_dir, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def plot(self):
        if not MATPLOT or len(self.metrics) < 2:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = [m['epoch'] for m in self.metrics]
        axes[0].plot(epochs, [m['train_loss'] for m in self.metrics], marker='o', label='Train Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        pre = [m.get('val_pre') for m in self.metrics]
        sen = [m.get('val_sen') for m in self.metrics]
        spe = [m.get('val_spe') for m in self.metrics]
        f1 = [m.get('val_f1') for m in self.metrics]
        if any(v is not None for v in f1):
            axes[1].plot(epochs, pre, marker='o', label='PRE')
            axes[1].plot(epochs, sen, marker='s', label='SEN')
            axes[1].plot(epochs, spe, marker='^', label='SPE')
            axes[1].plot(epochs, f1, marker='d', label='F1')
            axes[1].set_title('Validation Five Metrics (Weighted)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].set_ylim(0, 1.05)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.log_dir, 'five_metrics.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print('  Plot saved: %s' % path)


# ============================================================
# 7. 两阶段训练主函数
# ============================================================

def train_phase1(model, train_loader, valid_loader, criterion, device, args, wandb_run, logger, encoder):
    """
    Phase 1: 二分类预训练 (benign=0 vs attack=1)
    训练目标: 学"正常长什么样"
    """
    print()
    print('=' * 60)
    print('PHASE 1: Binary Pre-training (benign vs attack)')
    print('=' * 60)

    best_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.phase1_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = correct / total if total > 0 else 0
        lr = model.optimizer.param_groups[0]['lr']

        # W&B 记录
        if wandb_run is not None:
            wandb_run.log({
                'phase1/train/loss': train_loss,
                'phase1/train/accuracy': train_acc,
                'phase1/train/lr': lr,
                'phase1/train/epoch': epoch,
            }, step=epoch)

        # 验证
        eval_result = None
        if epoch % args.eval_every == 0 or epoch == args.phase1_epochs:
            eval_out = model.evaluate(valid_loader, device)
            eval_result = evaluate_complete_metrics(eval_out['labels'], eval_out['preds'], eval_out['probs'])
            print_metrics(eval_result, title='Phase 1 Epoch %d/%d Validation' % (epoch, args.phase1_epochs))

            # Phase 1 用 binary AUC 作为早停指标
            binary_labels = (eval_out['labels'] > 0).astype(int)
            try:
                binary_auc = roc_auc_score(binary_labels, 1 - eval_out['probs'][:, 0])
            except Exception:
                binary_auc = 0.0

            if wandb_run is not None:
                wandb_run.log({
                    'phase1/validation/binary_auc': binary_auc,
                    'phase1/validation/ACC': eval_result['ACC'],
                }, step=epoch)

            # 保存最佳（基于 binary AUC）
            if binary_auc > best_f1:
                best_f1 = binary_auc
                best_epoch = epoch
                best_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': model.optimizer.state_dict(),
                    'binary_auc': binary_auc,
                }
                print('  [BEST Phase1] Binary AUC=%.4f' % binary_auc)

        logger.log_epoch(epoch, train_loss, train_acc, eval_result, lr,
                         epoch_time=time.time() - epoch_start)
        print('Phase1 Epoch %3d/%d | Loss=%.4f Acc=%.4f | Time=%.1fs' % (
            epoch, args.phase1_epochs, train_loss, train_acc, time.time() - epoch_start))

    print()
    print('Phase 1 Complete. Best Binary AUC: %.4f (Epoch %d)' % (best_f1, best_epoch))
    return best_state


def train_phase2(model, train_loader, valid_loader, criterion, device, args, wandb_run, logger, encoder, phase1_state):
    """
    Phase 2: 三分类微调 (benign/suspicious/malicious)
    冻结 Encoder，只训练 ClassificationHead
    """
    print()
    print('=' * 60)
    print('PHASE 2: Three-class Fine-tuning')
    print('  - Loading Phase 1 encoder weights...')
    print('  - Freezing encoder...')
    print('=' * 60)

    # 加载 Phase 1 权重（跳过 shape 不匹配的层，即 classification_head）
    if phase1_state is not None:
        pretrained_dict = phase1_state['model_state']
        model_dict = model.state_dict()
        compatible_dict = {}
        skipped_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                skipped_keys.append(k)
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        print('  Loaded %d/%d layers from Phase 1' % (len(compatible_dict), len(pretrained_dict)))
        if skipped_keys:
            print('  Skipped (shape mismatch): %s' % ', '.join(skipped_keys[:5]))

    # 冻结 Encoder
    frozen, trainable = model.freeze_encoder(lr_head=args.phase2_lr)

    # W&B 记录 Phase 2 配置
    if wandb_run is not None:
        wandb_run.config.update({
            'phase2_lr': args.phase2_lr,
            'phase2_epochs': args.phase2_epochs,
            'frozen_params': frozen,
            'trainable_params': trainable,
        })

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.phase2_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = correct / total if total > 0 else 0
        lr = model.optimizer.param_groups[0]['lr']

        # W&B 记录
        if wandb_run is not None:
            wandb_run.log({
                'phase2/train/loss': train_loss,
                'phase2/train/accuracy': train_acc,
                'phase2/train/lr': lr,
                'phase2/train/epoch': epoch,
            }, step=args.phase1_epochs + epoch)

        # 验证
        eval_result = None
        if epoch % args.eval_every == 0 or epoch == args.phase2_epochs:
            eval_out = model.evaluate(valid_loader, device)
            eval_result = evaluate_complete_metrics(eval_out['labels'], eval_out['preds'], eval_out['probs'])
            print_metrics(eval_result, title='Phase 2 Epoch %d/%d Validation' % (epoch, args.phase2_epochs))

            if wandb_run is not None:
                wandb_run.log({
                    'phase2/validation/ACC': eval_result['ACC'],
                    'phase2/validation/AUC': eval_result['AUC'] if eval_result['AUC'] is not None else 0,
                    'phase2/validation/Macro_F1': eval_result['Macro']['F1'],
                    'phase2/validation/Weighted_F1': eval_result['Weighted']['F1'],
                }, step=args.phase1_epochs + epoch)

                log_confusion_matrix_to_wandb(
                    wandb_run, eval_result['confusion_matrix'],
                    ['benign', 'suspicious', 'malicious'], step=args.phase1_epochs + epoch
                )
                log_per_class_metrics_to_wandb(wandb_run, eval_result, prefix='phase2/validation', step=args.phase1_epochs + epoch)

            if eval_result['Weighted']['F1'] > best_f1:
                best_f1 = eval_result['Weighted']['F1']
                best_epoch = epoch
                best_path = os.path.join(args.save_dir, 'transcl_v2_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': model.optimizer.state_dict(),
                    'encoder': encoder,
                    'val_f1': best_f1,
                    'phase1_state': phase1_state,
                }, best_path)
                print('  [BEST Phase2] Saved (Weighted F1=%.4f)' % best_f1)

                if wandb_run is not None:
                    log_artifact_to_wandb(
                        wandb_run, best_path,
                        artifact_name='transcl-phase2-best-model',
                        artifact_type='model',
                        metadata={'epoch': epoch, 'val_f1': best_f1, 'val_acc': eval_result['ACC']}
                    )

        logger.log_epoch(args.phase1_epochs + epoch, train_loss, train_acc, eval_result, lr,
                         epoch_time=time.time() - epoch_start)
        print('Phase2 Epoch %3d/%d | Loss=%.4f Acc=%.4f | Time=%.1fs' % (
            epoch, args.phase2_epochs, train_loss, train_acc, time.time() - epoch_start))

    print()
    print('Phase 2 Complete. Best Weighted F1: %.4f (Epoch %d)' % (best_f1, best_epoch))
    return best_f1, best_epoch


# ============================================================
# 8. main 函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Trans-CL SOC Classifier v3 (Two-Stage)')
    parser.add_argument('--train-csv', required=True, help='训练集路径 (.csv 或 .parquet，含label或label_binary列)')
    parser.add_argument('--valid-csv', required=True, help='验证集路径 (.csv 或 .parquet，含label或label_binary列；无标签时会从训练集划分验证集)')
    # Phase 1 参数
    parser.add_argument('--phase1-epochs', type=int, default=15, help='Phase 1 二分类预训练轮数')
    parser.add_argument('--phase1-lr', type=float, default=1e-4, help='Phase 1 学习率')
    # Phase 2 参数
    parser.add_argument('--phase2-epochs', type=int, default=15, help='Phase 2 三分类微调轮数')
    parser.add_argument('--phase2-lr', type=float, default=1e-5, help='Phase 2 分类头学习率')
    # 通用参数
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--alpha-scale', type=float, default=1.0)
    parser.add_argument('--save-dir', default='models/transcl_v2')
    parser.add_argument('--eval-every', type=int, default=5)
    # W&B 参数
    parser.add_argument('--wandb-project', default='soc-transcl', help='W&B 项目名称')
    parser.add_argument('--wandb-entity', default=None, help='W&B 实体/用户名')
    parser.add_argument('--wandb-run-name', default=None, help='W&B 运行名称')
    parser.add_argument('--wandb-tags', default='', help='W&B 标签，逗号分隔')
    parser.add_argument('--wandb-offline', action='store_true', help='W&B 离线模式')
    parser.add_argument('--wandb-disabled', action='store_true', help='禁用 W&B')
    args = parser.parse_args()

    # 设置 W&B 模式
    wandb_mode = os.environ.get('WANDB_MODE', 'online')
    if args.wandb_disabled:
        wandb_mode = 'disabled'
    elif args.wandb_offline:
        wandb_mode = 'offline'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('Trans-CL SOC Classifier v3 (Two-Stage)')
    print('Device: %s' % device)
    if device.type == 'cuda':
        print('  GPU: %s' % torch.cuda.get_device_name(0))
    print('W&B mode: %s' % wandb_mode)
    print('=' * 60)

    # ========== W&B 初始化 ==========
    wandb_run = None
    if WANDB_AVAILABLE and not args.wandb_disabled:
        tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
        tags.append('two-stage')
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags,
            mode=wandb_mode,
            config={
                'train_csv': args.train_csv,
                'valid_csv': args.valid_csv,
                'phase1_epochs': args.phase1_epochs,
                'phase1_lr': args.phase1_lr,
                'phase2_epochs': args.phase2_epochs,
                'phase2_lr': args.phase2_lr,
                'batch_size': args.batch_size,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'focal_gamma': args.focal_gamma,
                'alpha_scale': args.alpha_scale,
                'device': str(device),
            }
        )
        print('[W&B] Run URL: %s' % wandb_run.url)
    else:
        print('[W&B] Disabled or not available')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.save_dir, 'exp_%s' % timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Step 1: 加载数据
    print()
    print('Step 1: Loading Data...')
    train_df = load_data(args.train_csv)
    valid_df = load_data(args.valid_csv)

    if 'label' not in train_df.columns:
        raise ValueError('Training data must contain a "label" or "label_binary" column.')

    # 若验证集没有标签（如 competition 的 valid_input），从训练集划分 10% 作为验证集
    if 'label' not in valid_df.columns:
        print()
        print('[WARNING] Validation data has no label column.')
        print('  Creating a stratified 90/10 train/validation split from the training data.')
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            train_df,
            test_size=0.1,
            stratify=train_df['label'],
            random_state=42,
        )
        print('  -> New train: %d rows | Valid: %d rows' % (len(train_df), len(valid_df)))

    # Step 2: 特征编码
    print()
    print('Step 2: Encoding Features...')
    encoder = SOCFeatureEncoder()
    train_features = encoder.fit_transform(train_df)
    valid_features = encoder.transform(valid_df)

    # 原始三分类标签
    train_labels_3 = train_df['label'].values.astype(np.int64)
    valid_labels_3 = valid_df['label'].values.astype(np.int64)

    # Phase 1 二分类标签: 0->0, 1->1, 2->1
    train_labels_2 = (train_labels_3 > 0).astype(np.int64)
    valid_labels_2 = (valid_labels_3 > 0).astype(np.int64)

    print('  Train features: %s' % str(train_features.shape))
    print('  Valid features: %s' % str(valid_features.shape))
    print('  Phase 1 labels (binary): Benign=%d, Attack=%d' % 
          ((train_labels_2 == 0).sum(), (train_labels_2 == 1).sum()))
    print('  Phase 2 labels (3-class): Benign=%d, Suspicious=%d, Malicious=%d' % 
          ((train_labels_3 == 0).sum(), (train_labels_3 == 1).sum(), (train_labels_3 == 2).sum()))

    # 更新 W&B config
    if wandb_run is not None:
        wandb_run.config.update({
            'feature_dim': SOCFeatureEncoder.NUM_FEATURES,
            'num_columns': SOCFeatureEncoder.NUM_COLUMNS,
            'cat_columns': SOCFeatureEncoder.CAT_COLUMNS,
            'class_distribution': {
                'benign': int((train_labels_3 == 0).sum()),
                'suspicious': int((train_labels_3 == 1).sum()),
                'malicious': int((train_labels_3 == 2).sum()),
            },
        })

    # 数据集
    from torch.utils.data import TensorDataset, DataLoader

    # Phase 1 数据集 (二分类)
    train_ds_p1 = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels_2).long()
    )
    valid_ds_p1 = TensorDataset(
        torch.from_numpy(valid_features).float(),
        torch.from_numpy(valid_labels_2).long()
    )
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader_p1 = DataLoader(valid_ds_p1, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Phase 2 数据集 (三分类)
    train_ds_p2 = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels_3).long()
    )
    valid_ds_p2 = TensorDataset(
        torch.from_numpy(valid_features).float(),
        torch.from_numpy(valid_labels_3).long()
    )
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader_p2 = DataLoader(valid_ds_p2, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 本地日志
    logger = ExperimentLogger(log_dir)
    logger.log_hyperparams({
        'train_csv': args.train_csv, 'valid_csv': args.valid_csv,
        'phase1_epochs': args.phase1_epochs, 'phase1_lr': args.phase1_lr,
        'phase2_epochs': args.phase2_epochs, 'phase2_lr': args.phase2_lr,
        'batch_size': args.batch_size, 'hidden_dim': args.hidden_dim,
        'dropout': args.dropout, 'focal_gamma': args.focal_gamma,
        'alpha_scale': args.alpha_scale,
        'feature_dim': SOCFeatureEncoder.NUM_FEATURES,
        'num_columns': SOCFeatureEncoder.NUM_COLUMNS,
        'device': str(device),
    })

    # ========================================================
    # Phase 1: 二分类预训练
    # ========================================================
    print()
    print('Step 3: Phase 1 - Building Binary Model...')
    model_p1 = TransCL_SOCClassifier(
        device=device,
        hidden_dim=args.hidden_dim,
        num_classes=2,  # 二分类
        num_layers=4,
        nhead=4,
        dropout=args.dropout,
        learning_rate=args.phase1_lr,
        num_columns=SOCFeatureEncoder.NUM_COLUMNS,
    )
    total = sum(p.numel() for p in model_p1.parameters())
    print('  Parameters: {:,d} ({:.2f}M)'.format(total, total / 1e6))

    # Phase 1 损失: 二分类 FocalLoss
    counts_p1 = np.bincount(train_labels_2)
    weights_p1 = np.array([counts_p1.sum() / (c * 2.0) if c > 0 else 1.0 for c in counts_p1])
    weights_p1 = np.minimum(weights_p1, 20.0)
    alpha_p1 = torch.tensor(weights_p1, dtype=torch.float32).to(device) * args.alpha_scale
    criterion_p1 = FocalLoss(alpha=alpha_p1, gamma=args.focal_gamma)
    print('  Phase 1 Loss: FocalLoss(alpha=%s, gamma=%.1f)' % (str(alpha_p1.cpu().numpy()), args.focal_gamma))

    if wandb_run is not None:
        wandb_run.watch(model_p1, log='all', log_freq=100)

    # 执行 Phase 1
    phase1_state = train_phase1(
        model_p1, train_loader_p1, valid_loader_p1, criterion_p1,
        device, args, wandb_run, logger, encoder
    )

    # 保存 Phase 1 最终模型
    phase1_path = os.path.join(args.save_dir, 'transcl_v2_phase1_final.pth')
    torch.save({
        'epoch': args.phase1_epochs,
        'model_state': model_p1.state_dict(),
        'encoder': encoder,
        'phase': 'phase1_binary',
    }, phase1_path)
    print('  Phase 1 model saved: %s' % phase1_path)
    if wandb_run is not None:
        log_artifact_to_wandb(wandb_run, phase1_path, 'transcl-phase1-model', 'model')

    # ========================================================
    # Phase 2: 三分类微调
    # ========================================================
    print()
    print('Step 4: Phase 2 - Building 3-class Model...')
    model_p2 = TransCL_SOCClassifier(
        device=device,
        hidden_dim=args.hidden_dim,
        num_classes=3,  # 三分类
        num_layers=4,
        nhead=4,
        dropout=args.dropout,
        learning_rate=args.phase2_lr,
        num_columns=SOCFeatureEncoder.NUM_COLUMNS,
    )
    total_p2 = sum(p.numel() for p in model_p2.parameters())
    print('  Parameters: {:,d} ({:.2f}M)'.format(total_p2, total_p2 / 1e6))

    # Phase 2 损失: 三分类 FocalLoss
    class_weights = compute_class_weights(train_labels_3).to(device)
    alpha_p2 = class_weights * args.alpha_scale
    criterion_p2 = FocalLoss(alpha=alpha_p2, gamma=args.focal_gamma)
    print('  Phase 2 Loss: FocalLoss(alpha=%s, gamma=%.1f)' % (str(alpha_p2.cpu().numpy()), args.focal_gamma))

    # 执行 Phase 2
    best_f1, best_epoch = train_phase2(
        model_p2, train_loader_p2, valid_loader_p2, criterion_p2,
        device, args, wandb_run, logger, encoder, phase1_state
    )

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'transcl_v2_final.pth')
    torch.save({
        'model_state': model_p2.state_dict(),
        'encoder': encoder,
        'phase1_state': phase1_state,
        'phase': 'phase2_final',
    }, final_path)
    if wandb_run is not None:
        log_artifact_to_wandb(
            wandb_run, final_path,
            artifact_name='transcl-final-model',
            artifact_type='model',
            metadata={'best_epoch': best_epoch, 'best_f1': best_f1}
        )

    # 总结
    print()
    print('=' * 60)
    print('All Training Complete!')
    print('  Best Phase2 Weighted F1: %.4f (Epoch %d)' % (best_f1, best_epoch))
    print('  Log dir: %s' % log_dir)
    logger.plot()
    logger.log_summary(best_f1, best_epoch)

    if wandb_run is not None:
        wandb_run.finish()
        print('[W&B] Run finished')


if __name__ == '__main__':
    main()
