#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1 / v2 外部验证：valid_input + 私榜答案 -> 指标 + 混淆矩阵。"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_transcl_v1 import (
    _read_table,
    _attach_label,
    TransCLContrastiveModel,
    evaluate_metrics,
    print_metrics,
)
from soc_feature_encoder_v2 import SOCFeatureEncoderV2

warnings.filterwarnings('ignore')

CLASSES = ['Benign', 'Suspicious', 'Malicious']
BASE = os.path.dirname(os.path.abspath(__file__))


def save_confusion_matrix(cm, path, title, subtitle):
    cm = np.array(cm, dtype=float)
    total = cm.sum()
    pct = cm / total * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=13, pad=10, loc='left', fontweight='bold')
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color='#555555')
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            ax.text(j, i, f'{val:,}\n({pct[i, j]:.1f}%)', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else '#222222',
                    fontsize=9, linespacing=1.35)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved CM: {path}')


def load_v1_encoder(path):
    class CU(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'train_transcl_v1'
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        return CU(f).load()


def load_v2_encoder(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_eval(name, encoder, model, df, device):
    print(f'\n===== {name} =====')
    print(f'  encoding {len(df):,} samples...')
    X = encoder.transform(df)
    y = df['label'].values.astype(np.int64)
    Xt = torch.from_numpy(X).float()

    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(Xt), 512):
            batch = Xt[i:i + 512].to(device)
            lg = model(batch)
            pr = F.softmax(lg, 1)
            preds.append(lg.argmax(1).cpu().numpy())
            probs.append(pr.cpu().numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)

    res = evaluate_metrics(y, preds, probs)
    print_metrics(res, f'{name} External Validation')
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('[1/3] Loading valid data + answer...')
    vi = _read_table(os.path.join(BASE, 'main', 'data', 'competition', 'valid_input.parquet'))
    ans = _read_table(os.path.join(BASE, 'main', 'data', 'competition', 'valid_answer_private.parquet'))
    vi = vi.merge(ans[['event_id', 'label_binary']], on='event_id', how='left')
    vi = _attach_label(vi)
    print(f'  valid rows: {len(vi):,} | labels: {vi.label.value_counts().sort_index().to_dict()}')

    configs = [
        {
            'name': 'v1 (34 features)',
            'encoder': os.path.join(BASE, 'models', 'transcl_v1', 'v1_5ep', 'encoder.pkl'),
            'model': os.path.join(BASE, 'models', 'transcl_v1', 'v1_5ep', 'final.pth'),
            'num_features': 34,
            'loader': load_v1_encoder,
            'out_prefix': 'external_valid_v1',
        },
        {
            'name': 'v2 (19 features)',
            'encoder': os.path.join(BASE, 'models', 'transcl_v2', 'v2_5ep', 'encoder.pkl'),
            'model': os.path.join(BASE, 'models', 'transcl_v2', 'v2_5ep', 'final.pth'),
            'num_features': 19,
            'loader': load_v2_encoder,
            'out_prefix': 'external_valid_v2',
        },
    ]

    for cfg in configs:
        print(f'\n[2/3] Loading {cfg["name"]} encoder/model...')
        encoder = cfg['loader'](cfg['encoder'])
        model = TransCLContrastiveModel(device=device, num_classes=3, num_features=cfg['num_features'])
        ckpt = torch.load(cfg['model'], map_location=device)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()

        res = run_eval(cfg['name'], encoder, model, vi, device)

        cm_path = os.path.join(BASE, cfg['out_prefix'] + '_cm.png')
        subtitle = (f"External validation | ACC={res['ACC']:.4f} | "
                    f"Macro F1={res['Macro']['F1']:.4f} | Weighted F1={res['Weighted']['F1']:.4f}")
        save_confusion_matrix(res['cm'], cm_path, f'{cfg["name"]} Confusion Matrix', subtitle)

        json_path = os.path.join(BASE, cfg['out_prefix'] + '_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=float)
        print(f'  saved metrics: {json_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
