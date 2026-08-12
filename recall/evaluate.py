# %%
import argparse
import numpy as np
import time

# %%
def recall(rs, N=10): 
    """
    例子：
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> recall(rs, N=1)
    0.333333
    >>> recall(rs, N=2)
    >>> 0.6666667
    >>> recall(rs, N=3)
    >>> 1.0
    """
    
    recall_flags = [np.sum(r[0:N]) for r in rs] 
                                                
    return np.mean(recall_flags) 
                                 

# %%
text2similar = {}

similar_text_pair = "recall_dataset/dev.csv"

with open(similar_text_pair, "r", encoding="utf-8") as f: 
    for line in f:
        text, similar_text = line.rstrip().split("\t")
        text2similar[text] = similar_text 

# %%
rs = [] 

recall_result_file = "recall_result_file/recall_result.txt"
recall_num = 50 

with open(recall_result_file, "r", encoding="utf-8") as f: 
    relevance_labels = []
    for index, line in enumerate(f): 
        if index % recall_num == 0 and index != 0: 
            rs.append(relevance_labels)
            relevance_labels = [] 

        query, recalled_text, cosine_sim = line.rstrip().split("\t") 

        if text2similar[query] == recalled_text: 
            relevance_labels.append(1) 
        else:
            relevance_labels.append(0)
        

# %%
recalls = [1, 5, 10, 20, 50] 
recall_N = [] 

for topN in recalls:
    R = round(100 * recall(rs, N=topN), 3) 
    recall_N.append(R) 

# %%
result_tsv_file = "recall_result_file/result.tsv"

with open(result_tsv_file, "w", encoding="utf-8") as f: 
    res = []

    for i in range(len(recalls)): 
        N = recalls[i] 
        recall_val = recall_N[i] 
        print("recall@{}={}".format(N, recall_val)) 
        res.append(str(recall_val)) 
    
    f.write("\t".join(res) + "\n") 

# %%



