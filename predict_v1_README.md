# Trans-CL SOC 预测脚本 v1（predict_transcl_v1.py）

本分支 `codex/predict-v1` 用于维护竞赛的预测脚本 v1 版本。它基于论文原版代码（`main` 分支）分出，完整保留论文 Trans-CL 的框架结构与全部论文代码，仅新增预测脚本 v1 与本文档。

## 一、相较于论文代码的改动

论文原版代码（commit `187c821`）是 Trans-CL 论文的原始实现，面向 CICIDS2017 数据集的网络流量入侵检测：输入 pcap 流量，经 `packetCollector.py` → `modelPipeline.py`（NFStreamer）→ `model.py`（BERT + NTXent 对比学习）→ `train.py` 完成训练。

本仓库在此基础上做了面向"第二届浙江省大学生人工智能竞赛·算法2：基于 SOC 日志网络安全威胁检测"的竞赛化改造，核心改动如下：

| 维度 | 论文原版 | 竞赛版本 |
| --- | --- | --- |
| 任务 | CICIDS2017 网络流量入侵检测 | SOC 日志三分类（benign / suspicious / malicious） |
| 输入数据 | pcap 流量文件 | parquet/csv 脱敏日志（event_id、timestamp、pipeline、src_ip、dst_ip、message_sanitized、label_binary 等） |
| 训练脚本 | train.py + model.py | 新增自包含 train_transcl_v4.py（32D 特征编码 + 对比学习 + 监督分类 + W&B） |
| 预测脚本 | 无（训练后直接评估） | predict_transcl_v4.py / predict_transcl_v1.py，输出 res.csv |
| 可解释性 | 无 | shap_analysis.py |
| 依赖 | 原 requirements.txt | 补充 torch、pandas、pyarrow、wandb、scikit-learn 等 |

### predict_transcl_v1.py 相对论文代码的关键点

1. 论文原版没有独立预测脚本，v1 新增独立推理流程：加载 `train_transcl_v4.py` 训练出的模型权重与特征编码器，对测试集批量推理。
2. 输出格式对齐赛题：`res.csv` 含 `event_id` 与 `pred_label` 两列，其中 `pred_label` 为字符串枚举 `benign` / `suspicious` / `malicious`（论文原版无此提交格式）。
3. 支持 W&B 记录推理指标（准确率、加权/宏平均 F1、混淆矩阵）。

## 二、用法

```powershell
# 1. 训练，得到 final.pth 与 encoder.pkl
python train_transcl_v4.py --data-path main/data/competition/train.parquet

# 2. 推理，输出符合赛题格式的 res.csv
python predict_transcl_v1.py `
  --data-path main/data/competition/valid_input.parquet `
  --model-path models/transcl_v4/final.pth `
  --encoder-path models/transcl_v4/encoder.pkl `
  --output-path res.csv
```

参数说明：

- `--data-path`：待预测的测试集（parquet/csv，须含 `event_id`）
- `--model-path`：训练保存的模型权重（`final.pth` 或 `best.pth`）
- `--encoder-path`：训练保存的特征编码器（`encoder.pkl`）
- `--output-path`：输出 csv 路径，默认 `res.csv`
- `--batch-size`：推理批大小，默认 512

## 三、文件说明

- `predict_transcl_v1.py`：预测脚本 v1（本分支新增）
- 论文原版代码（`model.py`、`train.py`、`modelPipeline.py`、`packetCollector.py`、`read_if.py`、`labelling.py`、`cicids2017_config.json` 等）：完整保留，未改动
- `train_transcl_v4.py`：竞赛训练脚本（v1 预测脚本的依赖）
- 其余竞赛文件（`predict_transcl_v2/v4/v5.py`、`shap_analysis.py`、`analysis/` 等）：保留，未改动
