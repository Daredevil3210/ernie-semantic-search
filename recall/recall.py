# %%
from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import hnswlib
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.utils.log import logger
import paddlenlp

from base_model import SemanticIndexBase
from data2 import convert_example, create_dataloader
from data2 import gen_id2corpus
from ann_util import build_index

# %%
paddle.set_device("cpu") 

# %%
# 下面开始构建模型并载入模型参数

# %%
model_name = "ernie-1.0"

pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(model_name)

# %%
model = SemanticIndexBase(pretrained_model, output_emb_size=256)

# %%
params_path = "model_param/model_180/model_state.pdparams" 

if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path) 
    model.set_dict(state_dict) 
    print("Loaded parameters from %s" % params_path) 
else:
    raise ValueError("Please set params_path with correct pretrained model file")

# %%
# 下面加载语料库文件，并利用语料库中的数据来构造ANN索引库

# %%
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)

# %%
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)

# %%
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result

# %%
corpus_file = "recall_dataset/corpus.csv" 

id2corpus = gen_id2corpus(corpus_file) 

# %%
print(type(id2corpus))

# %%
for i in range(10):
    print(id2corpus[i]) 

# %%
corpus_list = []
for idx, text in id2corpus.items():
    corpus_list.append({idx: text}) 

# %%
for i in range(10):
    print(corpus_list[i])

# %%
corpus_ds = MapDataset(corpus_list) 

# %%
print(type(corpus_ds))

# %%
for i in range(10):
    print(corpus_ds[i])

# %%
batch_sampler = paddle.io.BatchSampler(corpus_ds, batch_size=64, shuffle=False)

corpus_data_loader = paddle.io.DataLoader(dataset=corpus_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# %%
#如果需要从头构建索引，就运行下面这段代码。

output_emb_size = 256
hnsw_max_elements = 1000000 
hnsw_ef = 100 
hnsw_m = 100 

final_index = build_index(output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m, corpus_data_loader, model)

save_index_dir = "index_file" 
if not os.path.exists(save_index_dir):
    os.makedirs(save_index_dir)

save_index_path = os.path.join(save_index_dir, "final_index.bin") 
final_index.save_index(save_index_path)

# %%
'''
#如果有现成的索引文件final_index.bin,就运行这段代码
save_index_path = "index_file/final_index.bin"
output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size) 
final_index.load_index(save_index_path) 
'''

# %%
# 下面获取验证数据集中的所有query

# %%
def get_query_text(similar_text_pair_file): 
    querys = []
    with open(similar_text_pair_file, "r", encoding="utf-8") as f:
        for line in f:
            splited_line = line.rstrip().split("\t") 
            if len(splited_line) != 2: 
                continue

            if not splited_line[0] or not splited_line[1]: 
                continue

            querys.append({"text": splited_line[0]}) 

    return querys

# %%
similar_text_pair_file = "recall_dataset/dev.csv" 

query_list = get_query_text(similar_text_pair_file) 

# %%
print(type(query_list))

# %%
print(query_list)

# %%
query_ds = MapDataset(query_list) 

# %%
batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=64, shuffle=False)

query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# %%
query_embedding = model.get_semantic_embedding(query_data_loader) 

# %%
# 下面针对验证集中的query进行召回，生成召回结果文件

# %%
recall_result_dir = "recall_result_file" 
 
if not os.path.exists(recall_result_dir): 
    os.mkdir(recall_result_dir)

recall_result_file = "recall_result.txt" 

recall_result_file = os.path.join(recall_result_dir, recall_result_file) 

# %%
# 下面正式开始召回

with open(recall_result_file, "w", encoding="utf-8") as f: 
    for batch_index, batch_query_embedding in enumerate(query_embedding): 
        recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 

        batch_size = len(cosine_sims)

        for row_index in range(batch_size):
            text_index = 64 * batch_index + row_index 
            for idx, doc_idx in enumerate(recalled_idx[row_index]):
                f.write( "{}\t{}\t{}\n".format(query_list[text_index]["text"], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx] ) )


