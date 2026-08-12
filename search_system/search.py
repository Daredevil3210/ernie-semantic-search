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

from model import PairwiseMatching
from base_model import SemanticIndexBase
from data2 import convert_example, create_dataloader
from data import convert_pairwise_example
from data2 import gen_id2corpus
from ann_util import build_index
import copy

# %%
# 下面是准备工作

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
# 下面是召回的准备工作

# %%
corpus_file = "corpus.csv" 

id2corpus = gen_id2corpus(corpus_file) 

# %%
save_index_path = "final_index.bin" 
output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size) 
final_index.load_index(save_index_path) 

# %%
model_name = "ernie-1.0"

tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)

# %%
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)

# %%
pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(model_name)

model = SemanticIndexBase(pretrained_model, output_emb_size=256)

params_path = "recall_model_state.pdparams" 

if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path) 
    model.set_dict(state_dict) 
    print("Loaded parameters from %s" % params_path) 
else:
    raise ValueError("Please set params_path with correct pretrained model file")

# %%
#下面是排序的准备工作

# %%
@paddle.no_grad()
def predict(model, data_loader):
    model.eval()

    batch_probs = []
    for batch_data in data_loader:
        input_ids, token_type_ids = batch_data
        batch_prob = model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
        batch_probs.append(batch_prob)

    conca_batch_probs = np.concatenate(batch_probs, axis=0) 

    return conca_batch_probs

# %%
rank_model_name = "ernie-3.0-medium-zh"

rank_pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(rank_model_name)

rank_model = PairwiseMatching(rank_pretrained_model)

rank_params_path = "rank_model_state.pdparams" 

if rank_params_path and os.path.isfile(rank_params_path): 
    state_dict = paddle.load(rank_params_path)
    rank_model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")

# %%
rank_tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)

# %%
rank_trans_func = partial(convert_pairwise_example, tokenizer=rank_tokenizer, max_seq_length=64, phase="predict")

# %%
#下面开始进行搜索

# %%
input_query = input("请输入需要查询的文献")

# %%
querys = []
querys.append({"text": input_query})

# %%
print(querys)

# %%
query_ds = MapDataset(querys)

query_batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=1, shuffle=False)

query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=query_batch_sampler, collate_fn=batchify_fn, return_list=True)

# %%
query_embedding = model.get_semantic_embedding(query_data_loader) 

# %%
recall_data = []

# %%
# 下面正式开始召回

for batch_index, batch_query_embedding in enumerate(query_embedding): 
    recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50)                                                                           

    for idx, doc_idx in enumerate(recalled_idx[0]): 
        recall_data.append({"query": input_query, "title": id2corpus[doc_idx]})
        print( "{}\t{}\n".format(id2corpus[doc_idx], 1.0 - cosine_sims[0][idx] )                                                                         
        )

# %%
for i in recall_data:
    print(i)

# %%
rank_ds = MapDataset(recall_data)

# %%
rank_ds_copy = copy.deepcopy(rank_ds)

rank_batch_sampler = paddle.io.BatchSampler(rank_ds, batch_size=16, shuffle=False)

rank_data_loader = paddle.io.DataLoader(dataset=rank_ds.map(rank_trans_func), batch_sampler=rank_batch_sampler, collate_fn=batchify_fn, return_list=True)

# %%
y_probs = predict(rank_model, rank_data_loader)

# %%
rank_result = [] 

for idx, prob in enumerate(y_probs): 
    text_pair = rank_ds_copy[idx] 
    text_pair["pred_prob"] = prob[0] 
    rank_result.append(text_pair) 
    print(text_pair) 


# %%
rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)

for i in rank_result:
    print("{}\t{}".format(i['title'], i['pred_prob']))

# %%


# %%
# 下面是可以整体运行的代码

input_query = input("请输入需要查询的文献")

querys = []
querys.append({"text": input_query})

query_ds = MapDataset(querys)

query_data_loader = create_dataloader(
    query_ds, mode="predict", batch_size=1, batchify_fn=batchify_fn, trans_fn=trans_func
)

query_embedding = model.get_semantic_embedding(query_data_loader) 

recall_data = []

# 下面正式开始召回

for batch_index, batch_query_embedding in enumerate(query_embedding): 
    recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 
    for idx, doc_idx in enumerate(recalled_idx[0]): 
        recall_data.append({"query": input_query, "title": id2corpus[doc_idx], "sim": 1.0 - cosine_sims[0][idx]})
        print( "{}\t{}\n".format(id2corpus[doc_idx], 1.0 - cosine_sims[0][idx] ) 
        )

rank_ds = MapDataset(recall_data)


rank_ds_copy = copy.deepcopy(rank_ds)

rank_data_loader = create_dataloader(
    rank_ds_copy, mode="predict", batch_size=16, batchify_fn=batchify_fn, trans_fn=rank_trans_func
)

y_probs = predict(rank_model, rank_data_loader)

rank_result = []

for idx, prob in enumerate(y_probs):
    text_pair = rank_ds[idx]
    text_pair["pred_prob"] = prob[0]
    rank_result.append(text_pair)

rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)

for i in rank_result:
    print("{}\t{}".format(i['title'], i['pred_prob']))


