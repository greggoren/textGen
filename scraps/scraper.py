import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from time import time

def  get_count(idx,result,len):
    return sum([len-1-result[t][idx] for t in result])

def indexes(res):
    result = {}
    for i,test in enumerate(res):
        result[i]={}
        for k,j in enumerate(test):
            result[i][j]=k
    return result


a = {i:np.random.random() for i in range(150000)}
s = []
start = time()
for j in range(5):
    s.append(a)
res = []
start_sort = time()
for test in s:
    res.append(sorted(list(test.keys()),key=lambda x:(test[x],x),reverse=True))
print("sort took "+str(time()-start_sort))
borda_counts = {}
length = len(a)
start_index= time()
res_indexes = indexes(res)
print("indexing took "+str(time()-start_index))
start_count = time()
borda_counts = {idx:get_count(idx,res_indexes,length) for idx in a}
print("count took "+str(time()-start_count))
chosen_cand = max(list(borda_counts.keys()),key=lambda x:(borda_counts[x],x))
print("all took",str(time()-start))