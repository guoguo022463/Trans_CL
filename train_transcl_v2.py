#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v2 —— 19 维合并版特征 + 对比学习 + 三分类监督

与 v1 的区别仅在于特征编码器：把 34 维压缩到 19 维（合并/删除冗余字段）。
模型结构、训练流程、损失函数、评估指标均直接复用 train_transcl_v1.py。
"""
import os
import argparse
import pickle
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from soc_feature_encoder_v2 import SOCFeatureEncoderV2
from train_transcl_v1 import (
    _read_table,
    _attach_label,
    load_train_valid,
    FocalLoss,
    TransCLContrastiveModel,
    train_contrastive,
    train_supervised,
    evaluate_model,
    save_metrics_row,
    save_confusion_matrix,
    save_run_report,
    auto_run_name,
    WANDB_AVAILABLE,
)

warnings.filterwarnings('ignore')


def main():
    p = argparse.ArgumentParser(description='Trans-CL v2 — 19D merged features + contrastive + supervised')
    p.add_argument('--data-path', required=True, help='训练集 parquet/csv（含 label_binary）')
    p.add_argument('--val-path', default=None)
    p.add_argument('--val-answer', default=None)
    p.add_argument('--split', type=float, default=0.15)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--contrastive-epochs', type=int, default=10)
    p.add_argument('--contrastive-lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--supervised-epochs', type=int, default=15)
    p.add_argument('--supervised-lr', type=float, default=1e-4)
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--alpha', default='1,5,8')
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--save-dir', default='models/transcl_v2')
    p.add_argument('--run-name', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb-project', default='transcl-soc-v2')
    p.add_argument('--wandb-offline', action='store_true')
    p.add_argument('--wandb-disabled', action='store_true')
    args = p.parse_args()

    if bool(args.val_path) != bool(args.val_answer):
        p.error('--val-path 与 --val-answer 需同时提供')

    run_name = args.run_name or auto_run_name(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Run: {run_name}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run = None
    if WANDB_AVAILABLE and not args.wandb_disabled:
        import wandb
        mode = 'offline' if args.wandb_offline else 'online'
        run = wandb.init(project=args.wandb_project, name=run_name, mode=mode, config=vars(args))
        print(f'[W&B] {run.url}')

    # 1. 数据
    train_df, valid_df = load_train_valid(args.data_path, args.val_path, args.val_answer,
                                          args.split, args.seed)

    # 2. 19 维特征编码
    encoder = SOCFeatureEncoderV2()
    X_train = encoder.fit_transform(train_df)
    X_valid = encoder.transform(valid_df)
    y_train = train_df['label'].values.astype(np.int64)
    y_valid = valid_df['label'].values.astype(np.int64)
    print(f'  Features: train={X_train.shape}, valid={X_valid.shape}')

    # 3. DataLoader
    tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                       batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(TensorDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long()),
                       batch_size=args.batch_size, shuffle=False)

    # 4. 模型（num_features=19）
    model = TransCLContrastiveModel(
        device=device, hidden_dim=args.hidden_dim, num_classes=3,
        num_layers=args.num_layers, nhead=args.nhead, dropout=args.dropout,
        temperature=args.temperature, num_features=SOCFeatureEncoderV2.NUM_FEATURES,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total:,} ({total/1e6:.2f}M)')

    # 5. 阶段一：对比学习
    if args.contrastive_epochs > 0:
        train_contrastive(model, tr_ld, device, args.contrastive_epochs, args.contrastive_lr, run)
        torch.save({'model': model.state_dict(), 'phase': 'contrastive'},
                   os.path.join(args.save_dir, 'contrastive.pth'))

    # 6. 阶段二：监督分类（FocalLoss）
    if args.alpha:
        parts = args.alpha.split(',')
        if len(parts) != 3:
            p.error('--alpha 需 3 个值(benign,suspicious,malicious)')
        w = [float(x) for x in parts]
    else:
        cw = np.bincount(y_train, minlength=3).astype(float)
        cw = np.maximum(cw, 1.0)
        w = cw.sum() / (cw * 3.0)
        w = np.minimum(w, 20.0)
    alpha = torch.tensor(w, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    print(f'  FocalLoss alpha={alpha.cpu().numpy()}, gamma={args.focal_gamma}')

    best_f1, best_ep, best_res = train_supervised(
        model, tr_ld, va_ld, criterion, device,
        args.supervised_epochs, args.supervised_lr, args.weight_decay, args.patience,
        run, args.eval_every, args.save_dir, step_offset=args.contrastive_epochs,
    )

    # 7. 保存最终产物
    print('\n[4/4] Saving final model...')
    best_ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'), map_location='cpu')
    fp = os.path.join(args.save_dir, 'final.pth')
    torch.save({'model': best_ckpt['model'], 'best_ep': best_ep, 'best_f1': best_f1}, fp)
    ep = os.path.join(args.save_dir, 'encoder.pkl')
    with open(ep, 'wb') as f:
        pickle.dump(encoder, f)
    print(f'  Saved: {fp}')
    print(f'  Encoder: {ep}')

    report = {
        'run_name': run_name,
        'save_dir': args.save_dir,
        'best_epoch': best_ep,
        'best_val_macro_f1': best_f1,
        'feature_dim': SOCFeatureEncoderV2.NUM_FEATURES,
        'data': {'train': args.data_path, 'val': args.val_path, 'val_answer': args.val_answer},
        'config': vars(args),
        'best_metrics': best_res if best_res is not None else None,
    }
    save_run_report(os.path.join(args.save_dir, 'run_report.json'), report)
    print(f'  Saved report: {os.path.join(args.save_dir, "run_report.json")}')

    if run:
        run.finish()
        print('[W&B] Done')

    print('\nTraining Complete!')
    print(f'  Best Val Macro F1: {best_f1:.4f} (Epoch {best_ep})')


if __name__ == '__main__':
    main()
