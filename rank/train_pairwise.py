# %%
import argparse
import os
import random
import time
from functools import partial

import numpy as np
import paddle
import pandas as pd
from data import convert_pairwise_example
from model import PairwiseMatching
from tqdm import tqdm

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup

# %%
# 下面加载训练集和验证集

# %%
def read_train(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        flag = 0
        for line in f: 
            if flag != 0: 
                data = line.rstrip().split("\t") 
                if len(data) != 3: 
                    continue
                yield {'query': data[0], 'title': data[1], 'neg_title': data[2]} 
                                                                                   
            flag = 1

# %%
def read_dev(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        flag = 0
        for line in f: 
            if flag != 0: 
                data = line.rstrip().split("\t") 
                if len(data) != 3: 
                    continue
                yield {'query': data[0], 'title': data[1], 'label': data[2]} 
                                                                               
            flag = 1

# %%
train_file = "rank_dataset/train_pairwise.csv"
train_ds = load_dataset(read_train, data_path=train_file, lazy=False) 

dev_file = "rank_dataset/dev_pairwise.csv"
dev_ds = load_dataset(read_dev, data_path=dev_file, lazy=False) 

# %%
# 输出训练数据集的前3条数据看一下
for i in range(3):
    print(train_ds[i])

# %%
# 输出验证数据集的前3条数据看一下
for i in range(3):
    print(dev_ds[i])

# %%
model_name = "ernie-3.0-medium-zh"

tokenizer = AutoTokenizer.from_pretrained(model_name) 

# %%
trans_func_train = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="train")

trans_func_eval = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="eval")

# %%
#演示
example = train_ds[0]
print(example)

query, pos_title, neg_title = example["query"], example["title"], example["neg_title"]
print(query)
print(pos_title)
print(neg_title)
print('---------------------------------------------')

pos_inputs = tokenizer(text=query, text_pair=pos_title, max_seq_len=128) 
neg_inputs = tokenizer(text=query, text_pair=neg_title, max_seq_len=128) 

print(pos_inputs)
print(neg_inputs)
print('---------------------------------------------')

pos_input_ids = pos_inputs["input_ids"] 
pos_token_type_ids = pos_inputs["token_type_ids"] 

print(pos_input_ids)
print(pos_token_type_ids)
print('---------------------------------------------')

neg_input_ids = neg_inputs["input_ids"] 
neg_token_type_ids = neg_inputs["token_type_ids"] 
print(neg_input_ids)
print(neg_token_type_ids)
print('---------------------------------------------')

result = [pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids]
print(result)

# %%
def batchify_fn_train(samples): 
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result

# %%
def batchify_fn_eval(samples): 
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
        Stack(dtype="int64"), 
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result

# %%
batch_sampler_train = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func_train), batch_sampler=batch_sampler_train, collate_fn=batchify_fn_train, return_list=True)

# %%
batch_sampler_dev = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)

dev_data_loader = paddle.io.DataLoader(dataset=dev_ds.map(trans_func_eval), batch_sampler=batch_sampler_dev, collate_fn=batchify_fn_eval, return_list=True)

# %%
# 下面搭建模型，并开始训练

# %%
pretrained_model = AutoModel.from_pretrained(model_name) 

# %%
model = PairwiseMatching(pretrained_model, margin=0.1) 

# %%
epochs = 3 

num_training_steps = len(train_data_loader) * epochs 

lr_scheduler = LinearDecayWithWarmup(2E-5, num_training_steps, 0.0)

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
metric = paddle.metric.Auc() 

# %%
@paddle.no_grad() 
def evaluate(model, metric, data_loader): 
    model.eval() 
    metric.reset()
 
    for idx, batch in enumerate(data_loader): 
        input_ids, token_type_ids, labels = batch
        pos_probs = model.predict(input_ids=input_ids, token_type_ids=token_type_ids) 
        neg_probs = 1.0 - pos_probs 
        preds = np.concatenate((neg_probs, pos_probs), axis=1) 

        metric.update(preds=preds, labels=labels) 
        auc = metric.accumulate() 

    print("phase: dev, auc: {:.3}".format(auc)) 
    metric.reset()
    model.train()

# %%
# 下面正式开始训练模型

save_dir="model_param"
global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1): 
        pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids = batch

        loss = model(
            pos_input_ids=pos_input_ids,
            neg_input_ids=neg_input_ids,
            pos_token_type_ids=pos_token_type_ids,
            neg_token_type_ids=neg_token_type_ids,
        ) 

        global_step += 1

        if global_step % 10 == 0: 
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            evaluate(model, metric, dev_data_loader)
            
            save_path = os.path.join(save_dir, "model_%d" % global_step) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_param_path = os.path.join(save_path, "model_state.pdparams") 
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_path) 

# %%


# %%
# 下面是把保存好的模型参数文件加载到一个新的模型中，并进行性能测试的过程

model_2 = PairwiseMatching(pretrained_model, margin=0.1)

params_path = "model_param/model_400/model_state.pdparams"

state_dict = paddle.load(params_path)
model_2.set_dict(state_dict)

evaluate(model_2, metric, dev_data_loader)

# %%



