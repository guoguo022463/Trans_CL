#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v1 推理脚本 (三分类) — 预测输出符合赛题格式
======================================================
用法:
  python predict_transcl_v1.py --data-path valid_input.parquet --model-path models/transcl_v1/final.pth --encoder-path models/transcl_v1/encoder.pkl --output-path res.csv
"""
import torch, torch.nn.functional as F
import argparse, warnings, numpy as np, pandas as pd, pickle
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--encoder-path', required=True)
    p.add_argument('--output-path', default='res.csv')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--wandb-project', default='transcl-soc')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-run-name', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.data_path.endswith(('.parquet', '.pq')):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path, low_memory=False)
    print(f'Loaded {len(df):,} samples')

    # 用自定义 Unpickler 解决 __main__ 模块路径问题
    # encoder.pkl 是在 train_transcl_v1.py（__main__）中保存的，predict 脚本中 __main__ 不同
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'train_transcl_v1'
            return super().find_class(module, name)

    with open(args.encoder_path, 'rb') as f:
        encoder = CustomUnpickler(f).load()
    print(f'Encoder: {encoder.NUM_FEATURES}D')

    from train_transcl_v1 import TransCLContrastiveModel, evaluate_metrics, print_metrics
    model = TransCLContrastiveModel(device=device, num_classes=3)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device); model.eval()
    print(f'Model loaded: {args.model_path}')

    feats = encoder.transform(df)
    X = torch.from_numpy(feats).float()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            batch = X[i:i+args.batch_size].to(device)
            lg = model(batch); pr = F.softmax(lg, 1)
            preds.append(lg.argmax(1).cpu().numpy())
            probs.append(pr.cpu().numpy())
    preds = np.concatenate(preds); probs = np.concatenate(probs)

    # 输出（赛题要求 pred_label 为字符串枚举值 benign/suspicious/malicious）
    label_names = np.array(['benign', 'suspicious', 'malicious'])
    out = pd.DataFrame({
        'event_id': df.get('event_id', range(len(df))),
        'pred_label': label_names[preds]
    })
    out.to_csv(args.output_path, index=False)
    print(f'Output: {args.output_path}')
    print(f'  Benign: {(preds==0).sum():,} | Suspicious: {(preds==1).sum():,} | Malicious: {(preds==2).sum():,}')

    # 有标签则评估
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

        if WANDB_AVAILABLE:
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             name=args.wandb_run_name or f'infer_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                             job_type='inference')
            run.summary['test_acc'] = res['ACC']
            run.summary['test_f1_weighted'] = res['Weighted']['F1']
            run.summary['test_f1_macro'] = res['Macro']['F1']
            names = ['benign', 'suspicious', 'malicious']
            data = [[names[i], names[j], res['cm'][i][j]] for i in range(3) for j in range(3)]
            tbl = wandb.Table(columns=['true', 'pred', 'count'], data=data)
            run.log({'test/confusion_matrix': tbl})
            art = wandb.Artifact('predictions', type='dataset')
            art.add_file(args.output_path); run.log_artifact(art)
            run.finish(); print('[W&B] Logged')

if __name__ == '__main__':
    main()
