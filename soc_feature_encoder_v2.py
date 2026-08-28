#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOC 日志特征编码器 v2 —— 19 维合并版

基于 SHAP + 相关性分析，把 v1 的 34 维压缩到 19 维，合并/删除冗余字段：
  0  identity_missing   : dst_host 或 username 缺失（两个缺失标志合并，恶意类强信号）
  1  time_sin           : hour 循环编码 sin
  2  time_cos           : hour 循环编码 cos
  3  pipeline_idx       : pipeline 分类 id
  4  username_idx       : username 分类 id
  5  src_host_idx       : src_host 分类 id
  6  src_ip_idx         : src_ip 分类 id
  7  ip_is_private      : src_ip 是否私有地址
  8  ip_subnet_hash     : /24 子网 hash
  9  msg_length         : 消息长度 / 1000
  10 kw_malicious_count : 恶意关键词命中数
  11 kw_benign_count    : 良性关键词命中数
  12-18 tfidf_pca_0..6  : TF-IDF(500) -> PCA(7) 前 7 维

删除项：time_hour_norm(与sin冗余)、time_is_weekend/time_is_night(近零)、
ip_is_loopback(恒0)、msg_is_missing(被pipeline完全决定)、
dst_host_missing/username_missing(合并为identity_missing)、tfidf_pca_7..15(近零)。
"""
import re
import ipaddress
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


class SOCFeatureEncoderV2:
    NUM_FEATURES = 19
    NUM_COLS = list(range(19))
    CAT_COLS = [3, 4, 5, 6]  # pipeline, username, src_host, src_ip

    FEATURE_NAMES = [
        'identity_missing', 'time_sin', 'time_cos',
        'pipeline_idx', 'username_idx', 'src_host_idx', 'src_ip_idx',
        'ip_is_private', 'ip_subnet_hash',
        'msg_length', 'kw_malicious_count', 'kw_benign_count',
        'tfidf_pca_0', 'tfidf_pca_1', 'tfidf_pca_2', 'tfidf_pca_3',
        'tfidf_pca_4', 'tfidf_pca_5', 'tfidf_pca_6',
    ]

    def __init__(self):
        self.maps = {
            'pipeline': {'__MISSING__': 0, '__UNK__': 1},
            'username': {'__MISSING__': 0, '__UNK__': 1},
            'src_host': {'__MISSING__': 0, '__UNK__': 1},
            'src_ip': {'__MISSING__': 0, '__UNK__': 1},
        }
        self.mal_hard = [
            'unauthorized', 'malicious', 'escalation', 'injection', 'brute force',
            'exfiltration', 'attack', 'blocked', 'denied', 'failed', 'violation',
            'breach', 'compromised', 'suspicious', 'alert', 'critical', 'error',
            'anomaly', 'intrusion', 'unusual', 'scan', 'probe', 'drop', 'reject',
            'timeout', 'overflow', 'exploit',
        ]
        self.ben_hard = [
            'login successful', 'normal', 'logged out', 'restarted', 'success',
            'allowed', 'completed', 'authenticated', 'approved', 'established',
            'healthy', 'connected', 'verified', 'granted', 'permitted', 'acknowledged',
        ]
        self.mal_auto = []
        self.ben_auto = []
        self.tfidf = None
        self.pca = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df):
        print('[2/4] Fitting feature encoder (19D)...')
        for col in ['username', 'src_host', 'src_ip', 'pipeline']:
            if col not in df.columns:
                continue
            vals = df[col].fillna('').replace('', '__MISSING__').unique()
            m = self.maps[col]
            m.clear()
            m['__MISSING__'] = 0
            m['__UNK__'] = 1
            for i, v in enumerate(vals):
                if v not in ('__MISSING__', '__UNK__'):
                    m[v] = i + 2
        self._fit_keywords(df)
        self._fit_tfidf_pca(df)
        print('  Encoder fitted.')

    def _fit_keywords(self, df):
        if 'label' not in df.columns:
            return

        def tok(text):
            w = re.findall(r'[a-z]+', str(text).lower())
            sw = {
                'the', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can', 'need', 'dare',
                'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'between', 'under', 'and', 'but', 'or', 'yet',
                'so', 'if', 'because', 'although', 'though', 'while', 'where',
                'when', 'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
                'their', 'user', 'host', 'ip', 'port', 'id', 'name', 'time',
                'date', 'log', 'event', 'product', 'vendor',
            }
            return [x for x in w if len(x) > 3 and x not in sw]

        bcnt, scnt, mcnt = Counter(), Counter(), Counter()
        max_per_cls = 50000
        chunk_size = 5000
        for label_val, counter, name in [(0, bcnt, 'benign'), (1, scnt, 'suspicious'), (2, mcnt, 'malicious')]:
            count = 0
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk_labels = df.iloc[start:end]['label'].values
                mask = (chunk_labels == label_val)
                n_match = mask.sum()
                if n_match == 0:
                    continue
                chunk_texts = df.iloc[start:end]['_text']
                matched_indices = np.where(mask)[0]
                for idx_in_chunk in matched_indices:
                    counter.update(tok(chunk_texts.iloc[idx_in_chunk]))
                    count += 1
                    if count >= max_per_cls:
                        break
                del chunk_labels, mask, chunk_texts, matched_indices
                if count >= max_per_cls:
                    break
            print(f'  [{name:12s}] keywords from {count:,} samples')

        for w, c in mcnt.most_common(300):
            if c >= 10 and bcnt.get(w, 0) < c * 0.3 and scnt.get(w, 0) < c * 0.5:
                self.mal_auto.append(w)
            if len(self.mal_auto) >= 50:
                break
        for w, c in bcnt.most_common(300):
            if c >= 100 and mcnt.get(w, 0) < c * 0.05 and scnt.get(w, 0) < c * 0.1:
                self.ben_auto.append(w)
            if len(self.ben_auto) >= 50:
                break
        print(f'  Auto keywords: mal={len(self.mal_auto)}, ben={len(self.ben_auto)}')

    def _fit_tfidf_pca(self, df):
        if '_text' not in df.columns:
            return
        max_samples = 100000
        chunk_size = 10000
        collected = []
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk_texts = df.iloc[start:end]['_text']
            for msg in chunk_texts:
                collected.append(re.sub(r'[^a-z0-9\s]', ' ', str(msg).lower()))
            del chunk_texts
            if len(collected) >= max_samples:
                break
        if len(collected) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(collected), size=max_samples, replace=False)
            collected = [collected[i] for i in sorted(indices)]
        self.tfidf = TfidfVectorizer(
            max_features=500, min_df=5, max_df=0.8,
            ngram_range=(1, 2), stop_words='english',
        )
        tfidf_mat = self.tfidf.fit_transform(collected)
        self.pca = PCA(n_components=7)
        self.pca.fit(tfidf_mat.toarray())
        exp = sum(self.pca.explained_variance_ratio_)
        print(f'  TF-IDF vocab={len(self.tfidf.vocabulary_)} -> PCA(7) exp={exp:.1%} (from {len(collected):,} samples)')

    # ------------------------------------------------------------------
    # transform helpers
    # ------------------------------------------------------------------
    def _time(self, ts):
        try:
            if isinstance(ts, str):
                ts = float(ts)
            dt = datetime.fromtimestamp(ts)
            h = dt.hour
            return [h / 24.0, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                    1.0 if dt.weekday() >= 5 else 0.0,
                    1.0 if (h < 6 or h > 22) else 0.0]
        except Exception:
            return [0.0, 0.0, 1.0, 0.0, 0.0]

    def _ip(self, ip_str):
        try:
            ip = ipaddress.ip_address(str(ip_str))
            priv = 1.0 if ip.is_private else 0.0
            loop = 1.0 if ip.is_loopback else 0.0
            net = ipaddress.ip_network(str(ip) + '/24', strict=False)
            sub = hash(str(net.network_address)) % 1000 / 1000.0
            return priv, sub, loop
        except Exception:
            return 0.0, 0.0, 0.0

    def _msg(self, row):
        text = str(row.get('_text', '')) if pd.notna(row.get('_text')) else ''
        ml = min(len(text) / 1000.0, 5.0)
        tl = text.lower()
        km = sum(1 for k in self.mal_hard + self.mal_auto if k in tl)
        kb = sum(1 for k in self.ben_hard + self.ben_auto if k in tl)
        return ml, km, kb

    def _tfidf(self, text):
        if self.tfidf is None or self.pca is None:
            return [0.0] * 7
        m = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
        v = self.tfidf.transform([m])
        return self.pca.transform(v.toarray())[0].tolist()

    def _is_missing(self, val):
        if val is None:
            return True
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return False

    def _lookup(self, map_name, val):
        m = self.maps[map_name]
        if self._is_missing(val):
            key = '__MISSING__'
        else:
            s = str(val).strip()
            key = s if s else '__MISSING__'
        return m.get(key, m.get('__UNK__', 1))

    def _ensure_text(self, df):
        if '_text' in df.columns:
            return df
        if 'message_sanitized' in df.columns:
            df['_text'] = df['message_sanitized'].fillna('').astype(str)
        elif 'product_name' in df.columns and 'vendor_name' in df.columns:
            df['_text'] = (df['product_name'].fillna('') + ' ' + df['vendor_name'].fillna('')).str.strip()
        elif 'product_name' in df.columns:
            df['_text'] = df['product_name'].fillna('').astype(str)
        else:
            df['_text'] = ''
        return df

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, df):
        df = self._ensure_text(df)
        feats = []
        for _, row in df.iterrows():
            t = self._time(row.get('timestamp', 0))
            time_sin, time_cos = t[1], t[2]

            p = self._lookup('pipeline', row.get('pipeline'))
            u = self._lookup('username', row.get('username'))
            sh = self._lookup('src_host', row.get('src_host'))
            sip = self._lookup('src_ip', row.get('src_ip'))

            priv, sub, _loop = self._ip(row.get('src_ip', ''))
            ml, km, kb = self._msg(row)
            tf = self._tfidf(str(row.get('_text', '')))

            dh_miss = 1.0 if self._is_missing(row.get('dst_host')) else 0.0
            uname_miss = 1.0 if self._is_missing(row.get('username')) else 0.0
            identity_miss = 1.0 if (dh_miss or uname_miss) else 0.0

            f = [
                identity_miss, time_sin, time_cos,
                float(p), float(u), float(sh), float(sip),
                priv, sub,
                ml, float(km), float(kb),
            ] + tf
            feats.append(f)
        return np.array(feats, dtype=np.float32)

    def fit_transform(self, df):
        df = self._ensure_text(df)
        self.fit(df)
        return self.transform(df)
