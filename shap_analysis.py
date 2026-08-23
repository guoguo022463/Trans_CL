#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trans-CL v4 - SHAP 特征重要性分析与可视化
"""
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
    """根据 SOCFeatureEncoder 还原 32 维特征的具体名称"""
    time_feats = ['time_hour_norm', 'time_sin', 'time_cos', 'time_is_weekend', 'time_is_night']
    cat_feats = ['pipeline_idx', 'username_idx', 'src_host_idx', 'src_ip_idx']
    ip_feats = ['ip_is_private', 'ip_subnet_hash', 'ip_is_loopback']
    msg_feats = ['msg_length', 'msg_is_missing', 'kw_malicious_count', 'kw_benign_count']
    tfidf_feats = [f'tfidf_pca_{i}' for i in range(16)]
    return time_feats + cat_feats + ip_feats + msg_feats + tfidf_feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True, help='验证集或测试集数据路径')
    p.add_argument('--model-path', required=True, help='训练好的 best.pth 路径')
    p.add_argument('--encoder-path', required=True, help='训练好的 encoder.pkl 路径')
    p.add_argument('--num-background', type=int, default=200, help='用于计算基线的背景样本数')
    p.add_argument('--num-test', type=int, default=100, help='用于生成 SHAP 值的测试样本数')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. 加载数据
    if args.data_path.endswith(('.parquet', '.pq')):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path, low_memory=False)

    # 2. 解决 __main__ 模块路径问题并加载 Encoder
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'train_transcl_v4'
            return super().find_class(module, name)

    with open(args.encoder_path, 'rb') as f:
        encoder = CustomUnpickler(f).load()

    # 3. 加载模型
    from train_transcl_v4 import TransCLContrastiveModel
    model = TransCLContrastiveModel(device=device, num_classes=3)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # 4. 特征转换与采样 (SHAP 计算非常耗时，必须采样)
    print("正在编码特征...")
    df_sample = df.sample(n=min(len(df), args.num_background + args.num_test), random_state=42)
    feats = encoder.transform(df_sample)
    X_tensor = torch.from_numpy(feats).float().to(device)

    # 划分背景数据集 (用于提供期望值) 和测试数据集 (用于解释)
    X_background = X_tensor[:args.num_background]
    X_test = X_tensor[args.num_background: args.num_background + args.num_test]
    feature_names = get_feature_names()

    # 5. 构建 SHAP GradientExplainer
    print("正在计算 SHAP 值 (可能需要几分钟)...")
    # 对于复杂的 Transformer 结构，GradientExplainer 比 DeepExplainer 更稳定
    explainer = shap.GradientExplainer(model, X_background)
    shap_values = explainer.shap_values(X_test)

    # 6. 提取“恶意类 (Malicious)”的 SHAP 值进行可视化
    # shap_values 是一个列表，包含每个类别的 SHAP 值。索引 2 对应 malicious
    shap_values_malicious = shap_values[2]

    # 如果输出是 tuple，转换为 numpy
    if isinstance(shap_values_malicious, torch.Tensor):
        shap_values_malicious = shap_values_malicious.cpu().detach().numpy()
    X_test_np = X_test.cpu().numpy()

    # 7. 绘制 SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_malicious,
        X_test_np,
        feature_names=feature_names,
        show=False,
        title="SHAP Summary for Malicious Class"
    )
    plt.tight_layout()
    plt.savefig('shap_summary_malicious.png', dpi=300, bbox_inches='tight')
    print("SHAP 图表已保存为: shap_summary_malicious.png")


if __name__ == '__main__':
    main()
