#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v1 / v2 实验混淆矩阵图（PNG），数据取自各自 run_report.json。"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CLASSES = ['Benign', 'Suspicious', 'Malicious']


def load_cm(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['best_metrics']['cm'], dtype=int), data


def draw(cm, out_path, title, subtitle):
    cm = np.array(cm, dtype=float)
    total = cm.sum()
    pct = cm / total * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=13, pad=10, loc='left', fontweight='bold')
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color='#555555')

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            txt = f'{val:,}\n({pct[i, j]:.1f}%)'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else '#222222',
                    fontsize=9, linespacing=1.35)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out_path}')


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    v1_cm, v1 = load_cm(os.path.join(out_dir, 'models', 'transcl_v1', '0826_05', 'run_report.json'))
    v1_m = v1['best_metrics']
    draw(v1_cm, os.path.join(out_dir, 'confusion_matrix_v1.png'),
         'v1 Confusion Matrix (34 features)',
         f"Internal 85/15 validation | ACC={v1_m['ACC']:.4f} | Macro F1={v1_m['Macro']['F1']:.4f}")

    v2_cm, v2 = load_cm(os.path.join(out_dir, 'models', 'transcl_v2', 'v2_5ep', 'run_report.json'))
    v2_m = v2['best_metrics']
    draw(v2_cm, os.path.join(out_dir, 'confusion_matrix_v2.png'),
         'v2 Confusion Matrix (19 merged features)',
         f"Internal 85/15 validation | ACC={v2_m['ACC']:.4f} | Macro F1={v2_m['Macro']['F1']:.4f}")


if __name__ == '__main__':
    main()
