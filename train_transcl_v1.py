#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v1 — 三分类 (benign/suspicious/malicious) + 对比学习 + W&B
完全自包含版本: 不依赖 model.py / transcl_classifier.py
修复: OutputLayer 512维, encode用column_embeddings, 无split, chunk OOM
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, argparse, warnings, re, ipaddress, pickle
import numpy as np, pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================
# 0. 核心模型组件 (完全内嵌)
# ============================================================

class ColumnEmbedding(nn.Module):
    """单列 embedding: 数值->linear"""
    def __init__(self, num_embeddings=None, embedding_dim=16, is_numeric=False):
        super().__init__()
        self.is_numeric = is_numeric
        if is_numeric:
            self.layer = nn.Linear(1, embedding_dim)
        else:
            self.layer = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        if self.is_numeric:
            return self.layer(x.unsqueeze(-1))
        return self.layer(x.long())


class TransCL_SOCClassifier(nn.Module):
    """Trans-CL 编码器 + 分类头"""
    def __init__(self, device, hidden_dim=256, num_classes=3, num_layers=4, nhead=4,
                 dropout=0.1, num_columns=32):
        super().__init__()
        self.device = device
        self.num_columns = num_columns
        self.hidden_dim = hidden_dim
        self.embedding_dim = 16
        self.column_embeddings = nn.ModuleList([
            ColumnEmbedding(embedding_dim=self.embedding_dim, is_numeric=True)
            for _ in range(num_columns)
        ])
        input_dim = num_columns * self.embedding_dim  # 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.column_embeddings)]
        embedded = torch.stack(embedded, dim=1)  # (batch, 32, 16)
        embedded = embedded.view(embedded.size(0), 1, -1)  # (batch, 1, 512)
        encoded = self.encoder(embedded)
        return self.classification_head(encoded[:, 0, :])


class OutputLayer(nn.Module):
    """对比学习投影头"""
    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.layer(x), dim=1)


class NTXent(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        # L2 归一化投影向量，使相似度限定在 [-1, 1]，从根源避免 exp 溢出导致的 NaN
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=sim.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        labels = torch.arange(batch_size, device=sim.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        return self.criterion(sim, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ============================================================
# 1. 数据加载 (训练集 + 独立验证集)
# ============================================================

def _read_table(path):
    """读取 parquet/csv 表格"""
    if str(path).endswith(('.parquet', '.pq')):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    # 物化 pyarrow-backed 列（如 ArrowStringArray）为 numpy 数组，
    # 避免 train_test_split 等重排操作触发 pyarrow ChunkedArray.take 一次性大块分配导致 OOM
    for col in df.columns:
        arr = df[col].array
        if 'Arrow' in type(arr).__name__ or hasattr(arr, '_pa_array'):
            # 用 dtype='object' 的 Series 赋值，防止 pandas 自动推断回 pyarrow-backed str dtype
            df[col] = pd.Series(arr.to_numpy(), dtype='object')
    return df


def _attach_label(df):
    """把 label_binary/label 列映射成 0/1/2 的 label 列"""
    label_col = 'label_binary' if 'label_binary' in df.columns else 'label'
    assert label_col in df.columns, f'Data must have "label_binary" or "label" column. Got: {df.columns.tolist()}'
    label_map = {'benign': 0, 'suspicious': 1, 'malicious': 2}
    if df[label_col].dtype == object or str(df[label_col].dtype).startswith('str'):
        for v in df[label_col].unique():
            if v not in label_map:
                print(f'  [WARNING] Unknown label "{v}", mapping to 0 (benign)')
                label_map[v] = 0
        df['label'] = df[label_col].map(label_map).astype(np.int64)
    else:
        df['label'] = df[label_col].astype(np.int64)
    return df


def load_train_valid(data_path, val_path=None, val_answer=None, split=0.15, seed=42):
    """加载训练集与验证集。
    - 训练集: data_path（含标签）
    - 验证集优先级:
      1) val_path + val_answer（外部验证，按 event_id 对齐）
      2) split（从训练集内部分层切出 split 比例做验证，默认 0.15 即 85/15）
    """
    print(f'[1/4] Loading training data: {data_path}')
    train_df = _attach_label(_read_table(data_path))

    if val_path and val_answer:
        print(f'  Loading validation input: {val_path}')
        valid_df = _read_table(val_path)
        print(f'  Loading validation answer: {val_answer}')
        ans = _read_table(val_answer)
        if 'event_id' in valid_df.columns and 'event_id' in ans.columns:
            valid_df = valid_df.merge(ans[['event_id', 'label_binary']], on='event_id', how='left')
        else:
            valid_df['label_binary'] = ans['label_binary'].values
        valid_df = _attach_label(valid_df)
    elif split and 0 < split < 1:
        print(f'  内部划分: {split*100:.0f}% 验证 / {(1-split)*100:.0f}% 训练（分层）')
        train_df, valid_df = train_test_split(
            train_df, test_size=split, random_state=seed, stratify=train_df['label'])
    else:
        raise ValueError(
            '无效的数据划分：未提供外部验证(--val-path/--val-answer)，'
            '且 --split 不在 (0, 1) 范围内。'
            '请提供独立验证集，或使用内部划分比例（如 --split 0.15）。')

    print('\n  ====== Dataset Split ======')
    for name, d in [('Train', train_df), ('Valid', valid_df)]:
        n = len(d)
        print(f'  {name:6s}: {n:>10,}')
        for idx, name2 in [(0, 'Benign'), (1, 'Suspicious'), (2, 'Malicious')]:
            c = int((d['label'] == idx).sum())
            print(f'    {name2:12s}: {c:>8,}  ({c/n*100:>5.1f}%)')
    print()
    return train_df, valid_df

# ============================================================
# 2. 32D SOC 特征编码器 (chunk-based OOM修复)
# ============================================================

class SOCFeatureEncoder:
    NUM_FEATURES = 34
    NUM_COLS = list(range(34))
    CAT_COLS = [5, 6, 7, 8]

    def __init__(self):
        self.maps = {
            'pipeline': {'__MISSING__': 0, '__UNK__': 1},
            'username': {'__MISSING__': 0, '__UNK__': 1},
            'src_host': {'__MISSING__': 0, '__UNK__': 1},
            'src_ip': {'__MISSING__': 0, '__UNK__': 1},
        }
        self.mal_hard = ['unauthorized','malicious','escalation','injection','brute force',
            'exfiltration','attack','blocked','denied','failed','violation','breach',
            'compromised','suspicious','alert','critical','error','anomaly','intrusion',
            'unusual','scan','probe','drop','reject','timeout','overflow','exploit']
        self.ben_hard = ['login successful','normal','logged out','restarted','success',
            'allowed','completed','authenticated','approved','established','healthy',
            'connected','verified','granted','permitted','acknowledged']
        self.mal_auto = []
        self.ben_auto = []
        self.tfidf = None
        self.pca = None

    def fit(self, df):
        print('[2/4] Fitting feature encoder (32D)...')
        for col in ['username','src_host','src_ip','pipeline']:
            if col not in df.columns: continue
            vals = df[col].fillna('').replace('','__MISSING__').unique()
            m = self.maps[col]
            m.clear(); m['__MISSING__'] = 0; m['__UNK__'] = 1
            for i, v in enumerate(vals):
                if v not in ('__MISSING__','__UNK__'):
                    m[v] = i + 2
        self._fit_keywords(df)
        self._fit_tfidf_pca(df)
        print('  Encoder fitted.')

    def _fit_keywords(self, df):
        if 'label' not in df.columns: return
        def tok(text):
            w = re.findall(r'[a-z]+', str(text).lower())
            sw = {'the','a','is','are','was','were','be','been','being','have','has','had',
                  'do','does','did','will','would','could','should','may','might','must',
                  'can','need','dare','ought','used','to','of','in','for','on','with','at',
                  'by','from','as','into','through','during','before','after','above','below',
                  'between','under','and','but','or','yet','so','if','because','although',
                  'though','while','where','when','that','which','who','whom','whose','what',
                  'this','these','those','i','me','my','we','our','you','your','he','him',
                  'his','she','her','it','its','they','them','their','user','host','ip',
                  'port','id','name','time','date','log','event','product','vendor'}
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
                if n_match == 0: continue
                chunk_texts = df.iloc[start:end]['_text']
                matched_indices = np.where(mask)[0]
                for idx_in_chunk in matched_indices:
                    msg = chunk_texts.iloc[idx_in_chunk]
                    counter.update(tok(msg))
                    count += 1
                    if count >= max_per_cls: break
                del chunk_labels, mask, chunk_texts, matched_indices
                if count >= max_per_cls: break
            print(f'  [{name:12s}] keywords from {count:,} samples')
        for w, c in mcnt.most_common(300):
            if c >= 10 and bcnt.get(w,0) < c * 0.3 and scnt.get(w,0) < c * 0.5:
                self.mal_auto.append(w)
            if len(self.mal_auto) >= 50: break
        for w, c in bcnt.most_common(300):
            if c >= 100 and mcnt.get(w,0) < c * 0.05 and scnt.get(w,0) < c * 0.1:
                self.ben_auto.append(w)
            if len(self.ben_auto) >= 50: break
        print(f'  Auto keywords: mal={len(self.mal_auto)}, ben={len(self.ben_auto)}')

    def _fit_tfidf_pca(self, df):
        if '_text' not in df.columns: return
        max_samples = 100000
        chunk_size = 10000
        collected = []
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk_texts = df.iloc[start:end]['_text']
            for msg in chunk_texts:
                collected.append(re.sub(r'[^a-z0-9\s]', ' ', str(msg).lower()))
            del chunk_texts
            if len(collected) >= max_samples: break
        if len(collected) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(collected), size=max_samples, replace=False)
            collected = [collected[i] for i in sorted(indices)]
        self.tfidf = TfidfVectorizer(max_features=500, min_df=5, max_df=0.8,
                                      ngram_range=(1,2), stop_words='english')
        tfidf_mat = self.tfidf.fit_transform(collected)
        self.pca = PCA(n_components=16)
        self.pca.fit(tfidf_mat.toarray())
        exp = sum(self.pca.explained_variance_ratio_)
        print(f'  TF-IDF vocab={len(self.tfidf.vocabulary_)} -> PCA(16) exp={exp:.1%} (from {len(collected):,} samples)')

    def _time(self, ts):
        try:
            if isinstance(ts, str): ts = float(ts)
            dt = datetime.fromtimestamp(ts)
            h = dt.hour
            return [h/24.0, np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                    1.0 if dt.weekday()>=5 else 0.0,
                    1.0 if (h<6 or h>22) else 0.0]
        except: return [0.0,0.0,1.0,0.0,0.0]

    def _ip(self, ip_str):
        try:
            ip = ipaddress.ip_address(str(ip_str))
            priv = 1.0 if ip.is_private else 0.0
            loop = 1.0 if ip.is_loopback else 0.0
            net = ipaddress.ip_network(str(ip)+'/24', strict=False)
            sub = hash(str(net.network_address)) % 1000 / 1000.0
            return priv, sub, loop
        except: return 0.0, 0.0, 0.0

    def _msg(self, row):
        text = str(row.get('_text','')) if pd.notna(row.get('_text')) else ''
        ml = min(len(text)/1000.0, 5.0)
        miss = 1.0 if (pd.isna(row.get('_text')) or str(row.get('_text','')).strip()=='') else 0.0
        tl = text.lower()
        km = sum(1 for k in self.mal_hard+self.mal_auto if k in tl)
        kb = sum(1 for k in self.ben_hard+self.ben_auto if k in tl)
        return ml, miss, km, kb

    def _tfidf(self, text):
        if self.tfidf is None or self.pca is None: return [0.0]*16
        m = re.sub(r'[^a-z0-9\s]',' ',str(text).lower())
        v = self.tfidf.transform([m])
        return self.pca.transform(v.toarray())[0].tolist()

    def _is_missing(self, val):
        """判断字段值是否缺失（None 或 NaN）"""
        if val is None:
            return True
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return False

    def _lookup(self, map_name, val):
        """把字段值映射为 ID，正确处理 NaN/空值 -> __MISSING__（而非误判为 '__nan__' 或 '__UNK__'）"""
        m = self.maps[map_name]
        if self._is_missing(val):
            key = '__MISSING__'
        else:
            s = str(val).strip()
            key = s if s else '__MISSING__'
        return m.get(key, m.get('__UNK__', 1))

    def transform(self, df):
        df = self._ensure_text(df)
        feats = []
        for _, row in df.iterrows():
            t = self._time(row.get('timestamp',0))
            p = self._lookup('pipeline', row.get('pipeline'))
            u = self._lookup('username', row.get('username'))
            sh = self._lookup('src_host', row.get('src_host'))
            sip = self._lookup('src_ip', row.get('src_ip'))
            priv, sub, loop = self._ip(row.get('src_ip',''))
            ml, miss, km, kb = self._msg(row)
            tf = self._tfidf(str(row.get('_text','')))
            dh_miss = 1.0 if self._is_missing(row.get('dst_host')) else 0.0
            uname_miss = 1.0 if self._is_missing(row.get('username')) else 0.0
            f = [t[0],t[1],t[2],t[3],t[4], float(p),float(u),float(sh),float(sip),
                 priv, sub, loop, ml, miss, float(km), float(kb),
                 dh_miss, uname_miss] + tf
            feats.append(f)
        return np.array(feats, dtype=np.float32)

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

    def fit_transform(self, df):
        df = self._ensure_text(df)
        self.fit(df)
        return self.transform(df)

# ============================================================
# 3. 数据增强 & 评估指标
# ============================================================

class LogAugmenter:
    def __init__(self, noise=0.05, mask=0.1):
        self.noise, self.mask = noise, mask
    def __call__(self, x):
        x2 = x.clone()
        n = torch.zeros_like(x2)
        for i in range(x2.shape[1]):
            n[:, i] = 1.0
        x2 = x2 + torch.randn_like(x2) * self.noise * n
        m = torch.rand_like(x2) < self.mask
        return x2 * (~m).float()


def evaluate_metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds, labels=[0,1,2])
    acc = float((preds==labels).mean())
    names = ['Benign','Suspicious','Malicious']
    per = {}
    for c in range(3):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        tn = cm.sum() - tp - fp - fn
        eps = 1e-8
        tpr = tp/(tp+fn+eps); fpr = fp/(fp+tn+eps); tnr = tn/(tn+fp+eps); fnr = fn/(fn+tp+eps)
        pre = tp/(tp+fp+eps); sen = tpr
        f1 = 2*pre*sen/(pre+sen+eps)
        per[f'c{c}'] = {'TPR':tpr,'FPR':fpr,'TNR':tnr,'FNR':fnr,'PRE':pre,'SEN':sen,'F1':f1}

    macro = {k: float(np.mean([per[f'c{c}'][k] for c in range(3)])) for k in ['TPR','FPR','TNR','FNR','PRE','SEN','F1']}
    w = np.bincount(labels, minlength=3).astype(float); w = w / w.sum()
    weighted = {k: float(np.sum([w[c]*per[f'c{c}'][k] for c in range(3)])) for k in ['TPR','FPR','TNR','FNR','PRE','SEN','F1']}

    try:
        auc = float(roc_auc_score((labels>0).astype(int), 1-probs[:,0]))
    except: auc = None

    return {'cm': cm.tolist(), 'per': per, 'ACC': acc,
            'Macro': macro, 'Weighted': weighted, 'AUC': auc}


def print_metrics(res, title='Evaluation'):
    print(f'\n{"="*70}'); print(f'  {title}'); print(f'{"="*70}')
    cm = res['cm']; names = ['Benign','Suspicious','Malicious']
    print('  Confusion Matrix:')
    print('  %-12s %8s %8s %8s' % ('','Pred_B','Pred_S','Pred_M'))
    for i in range(3):
        print('  %-12s %8d %8d %8d' % ('True_'+names[i], cm[i][0], cm[i][1], cm[i][2]))
    print()
    print('  [二级指标] TPR / FPR / TNR / FNR')
    print('  %-12s %8s %8s %8s %8s' % ('Class','TPR','FPR','TNR','FNR'))
    print('  ' + '-'*50)
    for c in range(3):
        m = res['per'][f'c{c}']
        print('  %-12s %8.4f %8.4f %8.4f %8.4f' % (names[c], m['TPR'], m['FPR'], m['TNR'], m['FNR']))
    print('  ' + '-'*50)
    print('  %-12s %8.4f %8.4f %8.4f %8.4f' % ('Macro', res['Macro']['TPR'], res['Macro']['FPR'], res['Macro']['TNR'], res['Macro']['FNR']))
    print('  %-12s %8.4f %8.4f %8.4f %8.4f' % ('Weighted', res['Weighted']['TPR'], res['Weighted']['FPR'], res['Weighted']['TNR'], res['Weighted']['FNR']))
    print()
    print('  [三级指标] PRE / SEN / F1')
    print('  %-12s %8s %8s %8s' % ('Class','PRE','SEN','F1'))
    print('  ' + '-'*40)
    for c in range(3):
        m = res['per'][f'c{c}']
        print('  %-12s %8.4f %8.4f %8.4f' % (names[c], m['PRE'], m['SEN'], m['F1']))
    print('  ' + '-'*40)
    print('  %-12s %8.4f %8.4f %8.4f' % ('Macro', res['Macro']['PRE'], res['Macro']['SEN'], res['Macro']['F1']))
    print('  %-12s %8.4f %8.4f %8.4f' % ('Weighted', res['Weighted']['PRE'], res['Weighted']['SEN'], res['Weighted']['F1']))
    print(f'\n  ACC = {res["ACC"]:.4f}'); print(f'  AUC = {res["AUC"]}' if res['AUC'] else '  AUC = N/A')
    print('='*70)


def wandb_log_all(run, labels, preds, res, prefix="val", step=None):
    if run is None: return
    try:
        cm_plot = wandb.plot.confusion_matrix(
            y_true=labels, preds=preds,
            class_names=["benign", "suspicious", "malicious"],
            title=f"{prefix} Confusion Matrix"
        )
        run.log({f"{prefix}/confusion_matrix": cm_plot}, step=step)
    except Exception:
        pass
    for level in ["Macro", "Weighted"]:
        for metric in ["TPR", "FPR", "TNR", "FNR"]:
            run.log({f"{prefix}/{level}_{metric}": res[level][metric]}, step=step)
    for cls_name, short in [("c0", "benign"), ("c1", "suspicious"), ("c2", "malicious")]:
        for metric in ["TPR", "FPR", "TNR", "FNR"]:
            run.log({f"{prefix}/{short}_{metric}": res["per"][cls_name][metric]}, step=step)
    for level in ["Macro", "Weighted"]:
        for metric in ["PRE", "SEN", "F1"]:
            run.log({f"{prefix}/{level}_{metric}": res[level][metric]}, step=step)
    for cls_name, short in [("c0", "benign"), ("c1", "suspicious"), ("c2", "malicious")]:
        for metric in ["PRE", "SEN", "F1"]:
            run.log({f"{prefix}/{short}_{metric}": res["per"][cls_name][metric]}, step=step)
    run.log({f"{prefix}/ACC": res["ACC"], f"{prefix}/Macro_F1": res["Macro"]["F1"],
             f"{prefix}/Weighted_F1": res["Weighted"]["F1"]}, step=step)
    if res["AUC"] is not None:
        run.log({f"{prefix}/AUC": res["AUC"]}, step=step)

# ============================================================
# 4. 三分类对比学习模型 (Wrapper)
# ============================================================

class TransCLContrastiveModel(nn.Module):
    def __init__(self, device, hidden_dim=256, num_classes=3, num_layers=4, nhead=4,
                 dropout=0.1, temperature=0.5, num_features=None):
        super().__init__()
        self.device = device
        self.num_features = num_features or SOCFeatureEncoder.NUM_FEATURES
        self.base = TransCL_SOCClassifier(
            device=device, hidden_dim=hidden_dim, num_classes=num_classes,
            num_layers=num_layers, nhead=nhead, dropout=dropout,
            num_columns=self.num_features
        )
        encoder_output_dim = self.num_features * 16
        self.projection = OutputLayer(input_dim=encoder_output_dim, output_dim=128)
        self.ntxent = NTXent(temperature=temperature)
        self.to(device)

    def encode(self, x):
        x = x.to(self.device)
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.base.column_embeddings)]
        embedded = torch.stack(embedded, dim=1)  # (batch, 32, 16)
        embedded = embedded.view(embedded.size(0), 1, -1)  # (batch, 1, 512)
        enc = self.base.encoder(embedded)
        return enc[:, 0, :]  # (batch, 512)

    def forward_contrastive(self, x, x_aug):
        z1 = self.encode(x); z2 = self.encode(x_aug)
        p1 = self.projection(z1); p2 = self.projection(z2)
        return p1, p2

    def forward(self, x):
        z = self.encode(x)
        return self.base.classification_head(z)


# ============================================================
# 5. 训练流程
# ============================================================

def train_contrastive(model, train_loader, device, epochs, lr, run=None):
    print(); print('='*60); print('PHASE 1: Contrastive Learning (NTXent)'); print('='*60)
    aug = LogAugmenter(noise=0.05, mask=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train(); total_loss, n = 0.0, 0
        epoch_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        aborted = False
        for x, _ in train_loader:
            x = x.to(device)
            x_aug = aug(x)
            p1, p2 = model.forward_contrastive(x, x_aug)
            loss = model.ntxent(p1, p2)
            if not torch.isfinite(loss):
                print(f'  [WARNING] Contrastive Ep {ep}: loss={loss.item()} non-finite, skip batch')
                continue
            opt.zero_grad(); loss.backward()
            grads_ok = all(bool(torch.isfinite(p.grad).all())
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [WARNING] Contrastive Ep {ep}: gradient non-finite, skip batch')
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                print(f'  [ERROR] Contrastive Ep {ep}: weights became NaN/Inf, rollback & abort phase')
                model.load_state_dict(epoch_snapshot)
                aborted = True
                break
            total_loss += loss.item(); n += 1
        if aborted:
            print(f'  [ERROR] Contrastive phase aborted at Ep {ep} due to NaN/Inf weights')
            break
        avg = total_loss / n if n else float('nan')
        print(f'  Contrastive Ep {ep}/{epochs} | Loss={avg:.4f}')
        if not torch.isfinite(torch.tensor(avg)):
            print(f'  [WARNING] Contrastive Ep {ep}: all batches non-finite, weights may be NaN')
            break
        if run: run.log({'contrastive/loss': avg, 'contrastive/epoch': ep}, step=ep)

def train_supervised(model, train_loader, valid_loader, criterion, device, epochs, lr,
                     weight_decay=1e-5, patience=6, run=None, eval_every=1, save_dir='models/transcl_v1',
                     step_offset=0):
    print(); print('='*60); print('PHASE 2: Supervised Classification (3-class)'); print('='*60)
    os.makedirs(save_dir, exist_ok=True)
    for p in model.parameters(): p.requires_grad = False
    for p in model.base.classification_head.parameters(): p.requires_grad = True
    for p in model.projection.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=weight_decay)
    best_f1, best_ep = 0.0, 0
    best_res = None
    no_improve = 0
    for ep in range(1, epochs+1):
        model.train(); tot, corr, total = 0.0, 0, 0
        epoch_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                print(f'  [WARNING] Supervised Ep {ep}: loss={loss.item()} non-finite, skip batch')
                continue
            opt.zero_grad(); loss.backward()
            grads_ok = all(bool(torch.isfinite(p.grad).all())
                           for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                print(f'  [WARNING] Supervised Ep {ep}: gradient non-finite, skip batch')
                continue
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           max_norm=1.0)
            opt.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                print(f'  [ERROR] Supervised Ep {ep}: weights became NaN/Inf, rollback & abort')
                model.load_state_dict(epoch_snapshot)
                total = -1
                break
            tot += loss.item()
            corr += (logits.argmax(1) == y).sum().item(); total += len(y)
        if total <= 0:
            print(f'  [ERROR] Supervised Ep {ep}: no valid batches (total={total}), abort training')
            break
        acc = corr / total
        print(f'  Supervised Ep {ep}/{epochs} | Loss={tot/len(train_loader):.4f} Acc={acc:.4f}')
        if run: run.log({'supervised/train_loss': tot/len(train_loader), 'supervised/train_acc': acc}, step=step_offset+ep)
        if ep % eval_every == 0 or ep == epochs:
            res = evaluate_model(model, valid_loader, device, 'validation', run, step_offset+ep)
            # 选优/早停改用 Macro F1：对三个类别一视同仁，避免被多数类(benign)掩盖
            val_f1 = res['Macro']['F1']
            save_metrics_row(os.path.join(save_dir, 'metrics.csv'), ep,
                             tot/len(train_loader), acc, res)
            if val_f1 > best_f1:
                best_f1, best_ep = val_f1, ep
                best_res = res
                no_improve = 0
                p = os.path.join(save_dir, 'best.pth')
                torch.save({'epoch': ep, 'model': model.cpu().state_dict(), 'val_f1': best_f1}, p)
                model.to(device)
                save_confusion_matrix(res['cm'], os.path.join(save_dir, 'cm_best.png'),
                                      f'Confusion Matrix (Epoch {ep})')
                print(f'  [BEST] Saved (Macro F1={best_f1:.4f})')
            else:
                no_improve += 1
                print(f'  [INFO] No improve ({no_improve}/{patience})')
                if no_improve >= patience:
                    print(f'  [EARLY STOP] No improvement for {patience} epochs. Best: Ep {best_ep}')
                    break
    print(f'\nPhase 2 done. Best Val Macro F1: {best_f1:.4f} (Epoch {best_ep})')
    return best_f1, best_ep, best_res

def evaluate_model(model, loader, device, prefix='val', run=None, step=None):
    model.eval(); preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            lg = model(x); pr = F.softmax(lg, 1)
            preds.append(lg.argmax(1).cpu().numpy())
            labels.append(y.numpy()); probs.append(pr.cpu().numpy())
    preds = np.concatenate(preds); labels = np.concatenate(labels); probs = np.concatenate(probs)
    res = evaluate_metrics(labels, preds, probs)
    print_metrics(res, f'{prefix} (Epoch {step})' if step else prefix)
    if run:
        wandb_log_all(run, labels, preds, res, prefix, step=step)
    return res


def save_metrics_row(csv_path, epoch, train_loss, train_acc, res):
    """追加一行训练/验证指标到 CSV（首次写入表头）"""
    import csv as _csv
    header = ['epoch', 'train_loss', 'train_acc', 'val_acc', 'val_auc',
              'macro_tpr', 'macro_fpr', 'macro_tnr', 'macro_fnr',
              'macro_pre', 'macro_sen', 'macro_f1', 'weighted_f1',
              'benign_f1', 'suspicious_f1', 'malicious_f1']
    auc = res['AUC'] if res['AUC'] is not None else float('nan')
    row = [epoch, train_loss, train_acc, res['ACC'], auc,
           res['Macro']['TPR'], res['Macro']['FPR'], res['Macro']['TNR'], res['Macro']['FNR'],
           res['Macro']['PRE'], res['Macro']['SEN'], res['Macro']['F1'],
           res['Weighted']['F1'],
           res['per']['c0']['F1'], res['per']['c1']['F1'], res['per']['c2']['F1']]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def save_confusion_matrix(cm, path, title='Confusion Matrix'):
    """保存混淆矩阵图（PNG）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Benign', 'Suspicious', 'Malicious']
    cm = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_run_report(path, report):
    """保存训练报告 JSON"""
    import json as _json
    with open(path, 'w', encoding='utf-8') as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)


def auto_run_name(save_base):
    """生成 MMDD_XX 格式的实验名（当天第 XX 次）"""
    mdd = datetime.now().strftime('%m%d')
    count = 0
    if os.path.isdir(save_base):
        for name in os.listdir(save_base):
            if name.startswith(mdd + '_'):
                try:
                    n = int(name.rsplit('_', 1)[-1])
                    count = max(count, n)
                except ValueError:
                    pass
    return f'{mdd}_{count+1:02d}'


# ============================================================
# 6. main
# ============================================================

def main():
    p = argparse.ArgumentParser(description='Trans-CL v1 — 3-class + Contrastive + W&B (Self-contained)')
    p.add_argument('--data-path', required=True, help='训练集 parquet/csv（含 label_binary）')
    p.add_argument('--val-path', default=None, help='验证输入 parquet/csv（无标签，需配合 --val-answer）')
    p.add_argument('--val-answer', default=None, help='验证标签 parquet/csv（event_id + label_binary）')
    p.add_argument('--split', type=float, default=0.15,
                   help='从训练集内部分层切出的验证比例(0~1)，默认 0.15(85/15)；传 --val-path/--val-answer 时外部验证优先')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--contrastive-epochs', type=int, default=10)
    p.add_argument('--contrastive-lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--supervised-epochs', type=int, default=5)
    p.add_argument('--supervised-lr', type=float, default=1e-4)
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--alpha', default='1,5,8',
                   help='FocalLoss 类别权重，逗号分隔(benign,suspicious,malicious)，如 1,5,8；留空则用反频率自动权重')
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--wandb-project', default='transcl-soc')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-run-name', default=None)
    p.add_argument('--run-name', default=None,
                   help='实验名，默认自动生成 MMDD_XX（日期+当天次数）')
    p.add_argument('--wandb-tags', default='')
    p.add_argument('--wandb-offline', action='store_true',
                   help='离线模式（默认即离线，此参数为兼容保留）')
    p.add_argument('--wandb-online', action='store_true',
                   help='在线模式，同步到 wandb 云端（需网络稳定）')
    p.add_argument('--wandb-disabled', action='store_true')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--save-dir', default='models/transcl_v1')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    # 实验命名：默认 日期+次数（MMDD_XX），并作为 save_dir 子目录
    run_name = args.run_name or auto_run_name(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, run_name)

    mode = 'disabled' if args.wandb_disabled else ('online' if args.wandb_online else 'offline')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | W&B mode: {mode} | Run: {run_name}')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    run = None
    if WANDB_AVAILABLE and not args.wandb_disabled:
        tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
        tags.extend(['contrastive', 'parquet', '3class', '34d'])
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run_name or run_name,
                         tags=tags, mode=mode, config=vars(args))
        print(f'[W&B] {run.url}')

    # 1. Load data (训练集 + 独立验证集)
    if bool(args.val_path) != bool(args.val_answer):
        p.error('--val-path 与 --val-answer 需同时提供')
    train_df, valid_df = load_train_valid(args.data_path, args.val_path, args.val_answer,
                                          args.split, args.seed)

    # 2. Feature Encoding
    encoder = SOCFeatureEncoder()
    X_train = encoder.fit_transform(train_df)
    X_valid = encoder.transform(valid_df)
    y_train = train_df['label'].values.astype(np.int64)
    y_valid = valid_df['label'].values.astype(np.int64)
    print(f'  Features: train={X_train.shape}, valid={X_valid.shape}')

    # 3. DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                       batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(TensorDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long()),
                       batch_size=args.batch_size, shuffle=False)

    # 4. Model
    model = TransCLContrastiveModel(device, args.hidden_dim, 3, args.num_layers, args.nhead, args.dropout, args.temperature)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total:,} ({total/1e6:.2f}M)')
    if run: run.watch(model, log='all', log_freq=100)

    # 5. Phase 1: Contrastive
    os.makedirs(args.save_dir, exist_ok=True)
    if args.contrastive_epochs > 0:
        train_contrastive(model, tr_ld, device, args.contrastive_epochs, args.contrastive_lr, run)
        torch.save({'model': model.state_dict(), 'phase': 'contrastive'},
                   os.path.join(args.save_dir, 'contrastive.pth'))

    # 6. Phase 2: Supervised
    if args.alpha:
        parts = args.alpha.split(',')
        if len(parts) != 3:
            p.error(f'--alpha 需提供 3 个值(benign,suspicious,malicious)，当前: {args.alpha}')
        w = [float(x) for x in parts]
    else:
        cw = np.bincount(y_train, minlength=3).astype(float)
        cw = np.maximum(cw, 1.0); w = cw.sum() / (cw * 3.0); w = np.minimum(w, 20.0)
    alpha = torch.tensor(w, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    print(f'  FocalLoss alpha={alpha.cpu().numpy()}, gamma={args.focal_gamma}')
    best_f1, best_ep, best_res = train_supervised(model, tr_ld, va_ld, criterion, device,
                                                  args.supervised_epochs, args.supervised_lr,
                                                  args.weight_decay, args.patience,
                                                  run, args.eval_every, args.save_dir,
                                                  step_offset=args.contrastive_epochs)

    # 7. Final Save
    print(); print('[4/4] Saving final model (best weights)...')
    best_ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'), map_location='cpu')
    fp = os.path.join(args.save_dir, 'final.pth')
    torch.save({'model': best_ckpt['model'], 'best_ep': best_ep, 'best_f1': best_f1}, fp)
    ep = os.path.join(args.save_dir, 'encoder.pkl')
    with open(ep, 'wb') as f: pickle.dump(encoder, f)
    print(f'  Saved: {fp} (best epoch {best_ep})'); print(f'  Encoder: {ep}')

    if run:
        art = wandb.Artifact('model-files', type='model')
        art.add_file(fp); art.add_file(ep)
        run.log_artifact(art); run.finish()
        print('[W&B] Done')

    print(); print('='*70)
    print('Training Complete!')
    print(f'  Best Val Macro F1: {best_f1:.4f} (Epoch {best_ep})')

    # 8. 保存训练报告
    report = {
        'run_name': run_name,
        'save_dir': args.save_dir,
        'best_epoch': best_ep,
        'best_val_macro_f1': best_f1,
        'data': {
            'train': args.data_path,
            'val': args.val_path,
            'val_answer': args.val_answer,
        },
        'config': vars(args),
        'best_metrics': best_res if best_res is not None else None,
        'files': ['final.pth', 'encoder.pkl', 'best.pth', 'contrastive.pth',
                  'metrics.csv', 'cm_best.png'],
    }
    save_run_report(os.path.join(args.save_dir, 'run_report.json'), report)
    print(f'  Saved report: {os.path.join(args.save_dir, "run_report.json")}')
    print('='*70)

if __name__ == '__main__':
    main()
