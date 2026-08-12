# 基于 ERNIE 的语义检索与排序系统

这是一个基于 PaddlePaddle / PaddleNLP 实现的中文文献语义搜索项目。系统采用“两阶段检索”结构：先通过语义向量与 HNSW 索引召回候选文献，再使用 Pairwise Matching 模型进行精排。

## 项目组成

- `recall/`：召回模型微调、向量索引构建、召回测试与 Recall@N 评估。
- `rank/`：Pairwise 排序模型训练、预测与评估。
- `search_system/`：组合召回与排序模块的完整搜索流程。
- `环境配置命令/`：本机和服务器环境配置参考。
- `GPU服务器bug修复/`：GPU 服务器依赖导入问题的排查资料。

## 技术路线

1. 使用 ERNIE-1.0 将查询和文献编码为稠密向量。
2. 使用 HNSW 建立近似最近邻索引并召回候选文献。
3. 使用 ERNIE-3.0-medium-zh 训练 Pairwise Matching 模型。
4. 对召回结果计算相关性分数并重新排序。

## 环境要求

建议使用 Python 3.10，并在独立的 Conda 环境中运行。

```bash
conda create -n search python=3.10
conda activate search
pip install -r requirements.txt
```

`requirements.txt` 默认安装 CPU 版 PaddlePaddle。GPU 环境请根据 CUDA 版本，参考 `环境配置命令/环境配置命令-本机.txt` 中的命令安装对应版本。

## 使用方法

### 1. 训练召回模型

在 `recall/` 目录中按顺序运行：

```bash
python finetune.py
python recall.py
python evaluate.py
```

对应的 Notebook 为 `finetune.ipynb`、`recall.ipynb` 和 `evaluate.ipynb`。

### 2. 训练排序模型

在 `rank/` 目录中运行：

```bash
python train_pairwise.py
python predict_pairwise.py
```

### 3. 运行完整搜索流程

准备召回模型、排序模型和向量索引后，在 `search_system/` 目录运行：

```bash
python search.py
```

程序会提示输入待查询的文献内容，并输出召回结果及精排后的相关性分数。

## 数据与模型文件

GitHub 单文件大小有限制，因此仓库不包含以下本地生成或大体积文件：

- `*.pdparams`：训练后的模型参数。
- `*.bin`：HNSW 向量索引。
- `rank/rank_dataset/train_pairwise.csv`：完整 Pairwise 训练集。
- `recall/recall_dataset/corpus.csv`：与搜索模块重复的语料副本。

`search_system/corpus.csv` 保留为召回语料。若要单独运行 `recall/`，可将它复制到 `recall/recall_dataset/corpus.csv`，并根据脚本中的路径配置准备其余模型与索引文件。

## 目录说明

```text
.
├── recall/              # 语义召回训练与评估
├── rank/                # Pairwise 精排训练与预测
├── search_system/       # 端到端搜索系统
├── 环境配置命令/         # 环境安装参考
├── GPU服务器bug修复/     # GPU 环境问题排查资料
├── requirements.txt
└── README.md
```

## 说明

- 项目中的 `.py` 文件可直接运行，`.ipynb` 文件保留了实验过程。
- 首次运行 PaddleNLP 模型时需要联网下载预训练模型。
- 大文件和个人复试资料仅保留在本地，不会提交到公开仓库。
