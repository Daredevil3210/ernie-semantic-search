# %%
# 这个代码文件是把一个预训练模型ERNIE1.0构成的神经网络 利用in-batch-negative方法进行训练

# %%
import abc
import sys
import argparse
import os
import random
import time
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm 
from scipy.special import softmax
from scipy.special import expit
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import inference
import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.downloader import get_path_from_url
from visualdl import LogWriter
from data import convert_pairwise_example

# %%
paddle.set_device("cpu")

# %%
def read_text_pair(data_path): 
  with open(data_path, 'r', encoding='utf-8') as f: 
        for line in f: 
            data = line.rstrip().split("\t") 
            if len(data) != 2: 
                continue
            yield {'text_a': data[0], 'text_b': data[1]} 
                                                         


# %%

train_set_path='recall_dataset/train.csv' 
train_ds = load_dataset(read_text_pair, data_path=train_set_path, lazy=False) 

# %%
print(type(train_ds))

# %%

for i in range(3):
    print(train_ds[i])

# %%
# 下面开始构造训练数据的加载器

# %%
MODEL_NAME="ernie-1.0"

tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

# %%
def convert_example(example, tokenizer, max_seq_length=512): 
    result = []
    
    for key, text in example.items(): 
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"] 
        token_type_ids = encoded_inputs["token_type_ids"] 
        result += [input_ids, token_type_ids]

    return result 

# %%
#演示

a=[]
for key, text in train_ds[0].items():
    print(text)
    encoded_inputs = tokenizer(text=text, max_seq_len=512)
    print(encoded_inputs)
    print(encoded_inputs["input_ids"])
    print(encoded_inputs["token_type_ids"])
    a += [encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]]
    print('------------------------------------')
    
print(a)

# %%
from functools import partial

trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=64) 

# %%
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),   
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result

# %%
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=64, shuffle=True)
train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# %%
# 下面开始搭建召回模型

# %%
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# %%
from base_model import SemanticIndexBase

class SemanticIndexBatchNeg(SemanticIndexBase): 
    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        self.scale = scale

    def forward(self, query_input_ids,    title_input_ids,    query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,    title_token_type_ids=None, title_position_ids=None, title_attention_mask=None):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask) 

        title_cls_embedding = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)    
        
        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)  
        
        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype="float32") 

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        cosine_sim = cosine_sim * self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64') 
        labels = paddle.reshape(labels, shape=[-1, 1]) 

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss

# %%
model = SemanticIndexBatchNeg(pretrained_model, margin=0.1, scale=20, output_emb_size=256)

# %%
# 下面开始定义模型训练用到的各种参数，并进行模型训练

# %%
epochs=3 
num_training_steps = len(train_data_loader) * epochs 

# %%
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0) 

# %%
decay_params = [
        p.name for n, p in model.named_parameters() 
        if not any(nd in n for nd in ["bias", "norm"]) 
    ] 

# %%
optimizer = paddle.optimizer.AdamW( 
    learning_rate=lr_scheduler, 
    parameters=model.parameters(), 
    weight_decay=0.0, 
    apply_decay_param_fun=lambda x: x in decay_params)

# %%
save_dir='model_param'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

global_step = 0 
tic_train = time.time()

for epoch in range(1, epochs + 1): 
    for step, batch in enumerate(train_data_loader, start=1): 
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch 

        loss = model(query_input_ids=query_input_ids, title_input_ids=title_input_ids, query_token_type_ids=query_token_type_ids, title_token_type_ids=title_token_type_ids)

        global_step += 1 
        if global_step % 10 == 0: 
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train))) 
            tic_train = time.time() 

        loss.backward() 
        optimizer.step() 
        lr_scheduler.step() 
        optimizer.clear_grad() 

        if global_step % 10 == 0: 
            save_path = os.path.join(save_dir, "model_%d" % global_step)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_param_path = os.path.join(save_path, 'model_state.pdparams') 
            paddle.save(model.state_dict(), save_param_path) 
            tokenizer.save_pretrained(save_path) 


