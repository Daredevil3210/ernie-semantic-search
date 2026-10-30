[English](README.md) | [简体中文](README.zh.md)

# ERNIE 语义检索与排序

基于 PaddlePaddle/PaddleNLP 的中文文献检索学习项目：通过 ERNIE 稠密向量和 HNSW 索引召回候选标题，再使用 Pairwise 模型对查询与标题打分排序。

仓库包含实验 Notebook、Python 导出代码、辅助模块和部分数据集；训练参数、向量索引和完整排序训练集未纳入 Git。它不是开箱即用的托管搜索服务，复现需要准备模型产物并检查下述配置。

## 技术路线与学习范围

1. **语义召回**：使用批内负样本微调 `ernie-1.0`，编码为归一化的 256 维向量。
2. **候选检索**：使用 hnswlib 的内积距离检索 50 个候选。
3. **Pairwise 精排**：使用 Margin Ranking Loss 训练 `ernie-3.0-medium-zh`，并计算开发集 AUC。
4. **组合搜索**：输入查询、召回标题、计算相关性并排序。

```text
recall/              微调、索引、召回与 Recall@N
rank/                Pairwise 训练、AUC 评估与预测
search_system/       组合召回与排序的交互流程
环境配置命令/         原始本机/服务器环境配置说明
GPU服务器bug修复/     原始 GPU 依赖导入问题排查资料
```

`.ipynb` 保留实验流程，`.py` 对应文件包含按单元格导出的执行段。虽然部分文件导入参数解析模块，但并未提供可配置的命令行接口。

## 环境准备

原始配置说明使用 Python 3.10。依赖文件固定了 CPU 版 PaddlePaddle 2.6.2，但 PaddleNLP 等包未固定版本，因此并非完整可复现的环境锁定文件。

```bash
conda create -n search python=3.10
conda activate search
python -m pip install -r requirements.txt
```

使用 Notebook 还需在该环境中配置 Jupyter 与内核。首次调用 `from_pretrained` 需要下载模型或已有本地缓存。`环境配置命令/` 中的 GPU 安装命令为历史参考，应按本机情况选择兼容的 Paddle/CUDA 组合。部分脚本显式使用 `paddle.set_device("cpu")`，仅安装 GPU 包不会自动改变执行设备。

## 数据与模型前提

文件虽以 `.csv` 命名，文本对读取函数实际按 **Tab 分隔**：

- 召回训练/开发样本：查询和对应标题；`recall/finetune.py` 的读取函数要求两列，不处理表头。
- 召回语料：每行一个标题/文本，行顺序决定索引 ID。
- 排序训练：查询、正例标题、负例标题，读取时跳过首行表头。
- 排序开发：查询、标题、标签，同样跳过表头。
- 排序预测：读取函数要求三列，使用前两列作为查询和标题。

仓库跟踪了召回 train/dev/test、排序 dev/test，以及 `search_system/corpus.csv`，未跟踪：

```text
*.pdparams                              训练后的模型参数
*.bin                                   HNSW 索引
rank/rank_dataset/train_pairwise.csv     完整排序训练数据
recall/recall_dataset/corpus.csv         召回侧语料副本
```

使用已有语料时，将 `search_system/corpus.csv` 复制到 `recall/recall_dataset/corpus.csv`，索引与搜索期间必须保持行顺序一致。排序训练三元组需从有权使用的来源准备，仓库未提供下载或完整生成流程。`model_param/` 中的分词器词表和配置不等于训练后的神经网络权重。

## 运行流程

各阶段从对应目录运行，以保证相对导入和数据路径正确。分步解释可查看对应 Notebook。

### 1. 召回训练与评估

```bash
cd recall
python finetune.py
# 将 recall.py 的 params_path 改为上一步实际生成的权重路径。
python recall.py
python evaluate.py
```

召回前先准备语料副本。默认微调 3 轮，学习率 `5e-5`，每 10 步保存到 `model_param/model_<step>/model_state.pdparams`。召回默认读取 `model_180`，需确保该权重存在，或修改路径。

`recall.py` 构建 `index_file/final_index.bin`，将开发集查询的召回结果写入 `recall_result_file/recall_result.txt`。评估将 Recall@1/5/10/20/50 写入 `recall_result_file/result.tsv`。使用指标前应检查分组逻辑：当前评估循环没有追加最后一组候选。

### 2. 排序训练与预测

从仓库根目录执行：

```bash
cd rank
python train_pairwise.py
python predict_pairwise.py
```

运行前补齐排序训练文件。默认训练 3 轮，学习率 `2e-5`，Margin 为 0.1，每 100 步评估并保存一次。训练结尾评估和预测脚本都读取 `model_param/model_400/model_state.pdparams`，若实际训练得到其他步数的权重，需要修改这些路径。

### 3. 组合搜索

在 `search_system/` 下准备：

```text
corpus.csv
final_index.bin
recall_model_state.pdparams
rank_model_state.pdparams
```

使用彼此匹配的召回索引和召回权重，以及排序权重。运行前检查 `search_system/search.py`：当前 `rank_tokenizer` 使用召回的 `model_name`（`ernie-1.0`），而非 `rank_model_name`（`ernie-3.0-medium-zh`）。应先对齐分词器与排序模型，再判断结果有效性。导出代码还包含两段交互演示，可能提示输入两次。

处理上述前提后，从仓库根目录执行：

```bash
cd search_system
python search.py
```

输入文献查询，查看候选标题及精排后的分数。

## 验证状态与来源

仓库未提供自动化测试套件。本次文档更新仅检查源码和文件结构，未执行训练、加载权重或测量 Recall/AUC。历史输出和权重步数不构成独立验证的基准结果。

原有环境说明和问题排查资料予以保留，作为参考。数据、预训练模型和第三方资料仍受提供方条款约束。仓库未声明许可证文件或明确的开源许可授权。
