[English](README.md) | [简体中文](README.zh.md)

# ERNIE Semantic Search and Ranking

An educational Chinese literature-search pipeline built with PaddlePaddle/PaddleNLP. It retrieves candidate titles using dense ERNIE embeddings and an HNSW index, then scores query–title pairs with a pairwise ranker.

The repository contains experiment notebooks, Python exports, supporting modules, and some datasets. Trained parameters, vector indexes, and the complete ranking training set are excluded. It is not a ready-to-run hosted search service; reproduction requires preparing artifacts and reviewing the configuration points below.

## Architecture and learning scope

1. **Dense retrieval:** fine-tune `ernie-1.0` with in-batch negatives and encode text into normalized 256-dimensional embeddings.
2. **Candidate search:** use hnswlib with inner-product distance to retrieve 50 candidates.
3. **Pairwise ranking:** train `ernie-3.0-medium-zh` with a margin ranking loss and evaluate development AUC.
4. **Combined search:** accept a query, retrieve titles, score pairs, and sort by predicted relevance.

```text
recall/              Fine-tuning, indexing, retrieval, Recall@N
rank/                Pairwise training, AUC evaluation, prediction
search_system/       Combined interactive retrieval and ranking
环境配置命令/         Original local/server environment notes
GPU服务器bug修复/     Original GPU import troubleshooting materials
```

The `.ipynb` files retain the experiment workflow; their `.py` counterparts contain cell-style executable sections. They are not configurable command-line interfaces despite some imported argument-parsing modules.

## Environment

Original setup notes use Python 3.10. The dependency file pins CPU PaddlePaddle to 2.6.2 but leaves PaddleNLP and other packages unpinned, so it is not a fully reproducible lockfile.

```bash
conda create -n search python=3.10
conda activate search
python -m pip install -r requirements.txt
```

Notebook use additionally requires Jupyter and a kernel in this environment. First use of `from_pretrained` requires access to model downloads or an existing local cache. GPU setup commands in `环境配置命令/` are historical references; choose a compatible Paddle/CUDA installation for your machine. Several scripts explicitly call `paddle.set_device("cpu")`; installing a GPU package alone does not switch them to GPU.

## Data and artifact prerequisites

Although named `.csv`, the text-pair readers use **tab-separated** fields:

- Retrieval training/development pairs: query and matching title; the reader in `recall/finetune.py` expects two fields without header handling.
- Retrieval corpus: one title/text per line. Line order determines index IDs.
- Ranking training: query, positive title, negative title; the training reader skips the first line as a header.
- Ranking development: query, title, label; its reader also skips the header.
- Ranking prediction: the reader requires three fields and uses the first two as query/title.

The repository tracks retrieval train/dev/test files, ranking dev/test files, and `search_system/corpus.csv`. It does **not** track:

```text
*.pdparams                              Trained model parameters
*.bin                                   HNSW indexes
rank/rank_dataset/train_pairwise.csv     Complete ranking training data
recall/recall_dataset/corpus.csv         Retrieval-side corpus copy
```

To use the supplied corpus in retrieval, copy `search_system/corpus.csv` to `recall/recall_dataset/corpus.csv`. Keep its order unchanged between indexing and search. Supply ranking training triples from an authorized source; no download or complete generation procedure is provided. Tokenizer vocabulary/config files in `model_param/` are not trained neural weights.

## Workflow

Run each stage from its own directory so relative imports and data paths resolve. Review the matching notebook for the cell-by-cell explanation.

### 1. Train and evaluate retrieval

```bash
cd recall
python finetune.py
# Set recall.py's params_path to an actual checkpoint produced above.
python recall.py
python evaluate.py
```

Prepare the corpus copy before retrieval. Fine-tuning defaults to 3 epochs, learning rate `5e-5`, and checkpoint saving every 10 steps under `model_param/model_<step>/model_state.pdparams`. Retrieval defaults to `model_180`; that checkpoint must exist or the path must change.

`recall.py` builds `index_file/final_index.bin` and writes development-query candidates to `recall_result_file/recall_result.txt`. Evaluation writes Recall@1/5/10/20/50 to `recall_result_file/result.tsv`. Inspect grouping before relying on the metric: the current evaluation loop does not append its final candidate group.

### 2. Train and use the ranker

From the repository root:

```bash
cd rank
python train_pairwise.py
python predict_pairwise.py
```

Prepare the missing ranking training file first. Defaults are 3 epochs, learning rate `2e-5`, and margin 0.1. Training evaluates and saves every 100 steps. Its final evaluation and prediction script both use `model_param/model_400/model_state.pdparams`; update those paths if your run produces different checkpoint steps.

### 3. Run combined search

Prepare these files under `search_system/`:

```text
corpus.csv
final_index.bin
recall_model_state.pdparams
rank_model_state.pdparams
```

Use the retrieval index and its matching checkpoint, plus the ranking checkpoint. Review `search_system/search.py` before running: its `rank_tokenizer` currently loads the retrieval `model_name` (`ernie-1.0`) instead of `rank_model_name` (`ernie-3.0-medium-zh`). Align tokenizer and ranking model before treating results as valid. The export also contains two interactive demonstration sections and may prompt twice.

After resolving these prerequisites, run from the repository root:

```bash
cd search_system
python search.py
```

Enter a literature query to see retrieved titles and reranked scores.

## Validation, provenance, and rights

No automated test suite is provided. This documentation update inspected source and file structure without running training, loading weights, or measuring Recall/AUC. Archived outputs and checkpoint step numbers are not independently verified benchmark results.

Original environment notes and troubleshooting materials are retained as reference. Data, pretrained models, and third-party materials remain subject to their providers' terms. No repository license file or explicit open-source license grant is declared.
