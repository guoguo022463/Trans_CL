#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL SOC 推理脚本 v2 + W&B
生成 res.csv，可选上传预测结果到 W&B
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import csv
import time
import os
from datetime import datetime

from transcl_classifier import TransCL_SOCClassifier

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SOCFeatureEncoder:
    """SOC日志特征编码器（与训练脚本保持一致）"""
    def __init__(self):
        self.pipeline_map = {'__MISSING__': 0, '__UNK__': 1}
        self.username_map = {'__MISSING__': 0, '__UNK__': 1}
        self.src_host_map = {'__MISSING__': 0, '__UNK__': 1}
        self.src_ip_map = {'__MISSING__': 0, '__UNK__': 1}
        self.malicious_kws = [
            'unauthorized', 'malicious', 'escalation', 'injection',
            'brute force', 'exfiltration', 'attack', 'blocked',
            'denied', 'failed', 'violation', 'breach', 'compromised',
            'suspicious', 'alert', 'critical', 'error', 'anomaly'
        ]
        self.benign_kws = [
            'login successful', 'normal', 'logged out', 'restarted',
            'success', 'allowed', 'completed', 'authenticated',
            'approved', 'established'
        ]

    def _safe_get(self, row, col):
        val = row.get(col, '')
        if pd.isna(val):
            return '__MISSING__'
        val = str(val).strip()
        return val if val != '' else '__MISSING__'

    def transform_single(self, row):
        ts_val = row.get('timestamp', 0)
        try:
            if isinstance(ts_val, str):
                ts_val = float(ts_val)
            hour = datetime.fromtimestamp(ts_val).hour / 24.0
        except Exception:
            hour = 0.0
        pipeline = self._safe_get(row, 'pipeline')
        pipeline_id = self.pipeline_map.get(pipeline, self.pipeline_map.get('__UNK__', 1))
        username = self._safe_get(row, 'username')
        username_id = self.username_map.get(username, self.username_map.get('__UNK__', 1))
        src_host = self._safe_get(row, 'src_host')
        src_host_id = self.src_host_map.get(src_host, self.src_host_map.get('__UNK__', 1))
        src_ip = self._safe_get(row, 'src_ip')
        src_ip_id = self.src_ip_map.get(src_ip, self.src_ip_map.get('__UNK__', 1))
        msg = str(row.get('message_sanitized', '')) if pd.notna(row.get('message_sanitized')) else ''
        msg_len = min(len(msg) / 1000.0, 5.0)
        msg_missing = 1 if (pd.isna(row.get('message_sanitized')) or str(row.get('message_sanitized', '')).strip() == '') else 0
        msg_lower = msg.lower()
        kw_malicious = sum(1 for kw in self.malicious_kws if kw in msg_lower)
        kw_benign = sum(1 for kw in self.benign_kws if kw in msg_lower)
        return np.array([hour, pipeline_id, username_id, src_host_id, src_ip_id,
                         msg_len, msg_missing, kw_malicious, kw_benign], dtype=np.float32)

    def transform(self, df):
        features = np.stack([self.transform_single(row) for _, row in df.iterrows()])
        return features


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder = ckpt['encoder']
    model = TransCL_SOCClassifier(
        device=device, hidden_dim=256, num_classes=3,
        num_layers=4, nhead=4, dropout=0.1,
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, encoder


def predict(model, test_loader, device):
    all_preds = []
    all_event_ids = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, eids = batch
            else:
                x = batch[0]
                eids = batch[1] if len(batch) > 1 else None
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            if eids is not None:
                all_event_ids.extend(eids)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    return preds, probs, all_event_ids


def save_res_csv(event_ids, preds, output_path='res.csv'):
    label_map = {0: 'benign', 1: 'suspicious', 2: 'malicious'}
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['event_id', 'pred_label'])
        for eid, pred in zip(event_ids, preds):
            writer.writerow([str(eid), label_map[int(pred)]])
    print('Saved: %s (%d rows)' % (output_path, len(event_ids)))
    counts = np.bincount(preds, minlength=3)
    print('Prediction distribution:')
    print('  Benign:      %d (%.2f%%)' % (counts[0], counts[0]/len(preds)*100))
    print('  Suspicious:  %d (%.2f%%)' % (counts[1], counts[1]/len(preds)*100))
    print('  Malicious:   %d (%.2f%%)' % (counts[2], counts[2]/len(preds)*100))
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True, help='测试集CSV路径')
    parser.add_argument('--model-path', required=True, help='模型检查点路径')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--output', default='res.csv')
    parser.add_argument('--save-probs', action='store_true')
    # W&B 参数
    parser.add_argument('--wandb-project', default='soc-transcl', help='W&B 项目名称')
    parser.add_argument('--wandb-entity', default=None, help='W&B 实体/用户名')
    parser.add_argument('--wandb-run-name', default=None, help='W&B 运行名称')
    parser.add_argument('--wandb-disabled', action='store_true', help='禁用 W&B')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' % device)

    # W&B 初始化（推理模式）
    wandb_run = None
    if WANDB_AVAILABLE and not args.wandb_disabled:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or ('inference_%s' % datetime.now().strftime('%Y%m%d_%H%M%S')),
            job_type='inference',
            config={'test_csv': args.test_csv, 'model_path': args.model_path, 'batch_size': args.batch_size}
        )
        print('[W&B] Run URL: %s' % wandb_run.url)

    print('Loading model from %s...' % args.model_path)
    model, encoder = load_model(args.model_path, device)
    print('Model loaded.')

    print('Loading test data from %s...' % args.test_csv)
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    print('Test samples: %d' % len(test_df))

    if 'event_id' not in test_df.columns:
        raise ValueError('test_csv must contain "event_id" column')
    event_ids = test_df['event_id'].values

    print('Encoding features...')
    features = encoder.transform(test_df)
    print('Features shape: %s' % str(features.shape))

    from torch.utils.data import Dataset, DataLoader
    class EventDataset(Dataset):
        def __init__(self, features, event_ids):
            self.features = torch.from_numpy(features).float()
            self.event_ids = event_ids
        def __len__(self):
            return len(self.features)
        def __getitem__(self, idx):
            return self.features[idx], str(self.event_ids[idx])

    test_dataset = EventDataset(features, event_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print('Predicting...')
    start = time.time()
    preds, probs, returned_ids = predict(model, test_loader, device)
    elapsed = time.time() - start
    print('Prediction done in %.2fs (%.1f samples/sec)' % (elapsed, len(returned_ids)/elapsed))

    counts = save_res_csv(returned_ids, preds, args.output)

    # 上传预测结果到 W&B
    if wandb_run is not None:
        # 记录预测分布
        wandb_run.log({
            'inference/total_samples': len(returned_ids),
            'inference/benign_count': int(counts[0]),
            'inference/suspicious_count': int(counts[1]),
            'inference/malicious_count': int(counts[2]),
            'inference/benign_pct': float(counts[0] / len(returned_ids) * 100),
            'inference/suspicious_pct': float(counts[1] / len(returned_ids) * 100),
            'inference/malicious_pct': float(counts[2] / len(returned_ids) * 100),
            'inference/speed_samples_per_sec': len(returned_ids) / elapsed,
        })
        # 上传 res.csv 作为 artifact
        artifact = wandb.Artifact(name='predictions_%s' % wandb_run.id, type='predictions')
        artifact.add_file(args.output)
        wandb_run.log_artifact(artifact)
        print('[W&B] Predictions uploaded')

    if args.save_probs:
        prob_path = args.output.replace('.csv', '_probs.csv')
        with open(prob_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['event_id', 'prob_benign', 'prob_suspicious', 'prob_malicious'])
            for eid, p in zip(returned_ids, probs):
                writer.writerow([str(eid), '%.6f' % p[0], '%.6f' % p[1], '%.6f' % p[2]])
        print('Probabilities saved: %s' % prob_path)
        if wandb_run is not None:
            artifact = wandb.Artifact(name='probabilities_%s' % wandb_run.id, type='predictions')
            artifact.add_file(prob_path)
            wandb_run.log_artifact(artifact)

    print()
    print('Done! Submit %s to competition.' % args.output)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
