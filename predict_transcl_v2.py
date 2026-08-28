#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v2 推理脚本（19 维合并版）—— 输出符合赛题格式的 res.csv

用法:
  python predict_transcl_v2.py --data-path valid_input.parquet \
    --model-path models/transcl_v2/final.pth \
    --encoder-path models/transcl_v2/encoder.pkl --output-path res.csv
"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from soc_feature_encoder_v2 import SOCFeatureEncoderV2
from train_transcl_v1 import TransCLContrastiveModel, evaluate_metrics, print_metrics

warnings.filterwarnings('ignore')


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--encoder-path', required=True)
    p.add_argument('--output-path', default='res.csv')
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = read_table(args.data_path)
    print(f'Loaded {len(df):,} samples')

    with open(args.encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print(f'Encoder: {encoder.NUM_FEATURES}D')

    model = TransCLContrastiveModel(
        device=device, num_classes=3, num_features=SOCFeatureEncoderV2.NUM_FEATURES,
    )
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f'Model loaded: {args.model_path}')

    feats = encoder.transform(df)
    X = torch.from_numpy(feats).float()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            batch = X[i:i + args.batch_size].to(device)
            lg = model(batch)
            pr = F.softmax(lg, 1)
            preds.append(lg.argmax(1).cpu().numpy())
            probs.append(pr.cpu().numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)

    label_names = np.array(['benign', 'suspicious', 'malicious'])
    out = pd.DataFrame({
        'event_id': df.get('event_id', range(len(df))),
        'pred_label': label_names[preds],
    })
    out.to_csv(args.output_path, index=False)
    print(f'Output: {args.output_path}')
    print(f'  Benign: {(preds==0).sum():,} | Suspicious: {(preds==1).sum():,} | Malicious: {(preds==2).sum():,}')

    label_col = None
    if 'label_binary' in df.columns:
        label_col = 'label_binary'
    elif 'label' in df.columns:
        label_col = 'label'
    if label_col:
        label_map = {'benign': 0, 'suspicious': 1, 'malicious': 2}
        if df[label_col].dtype == object or str(df[label_col].dtype).startswith('str'):
            labels = df[label_col].map(label_map).fillna(0).values.astype(int)
        else:
            labels = df[label_col].values.astype(int)
        res = evaluate_metrics(labels, preds, probs)
        print_metrics(res, 'Inference Results')


if __name__ == '__main__':
    main()
