import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from time import time



d  = [i for i in range(1000000)]
start = time()
d.index(500564)
print("took",str(time()-start))
a = {i:np.random.random() for i in range(1000000)}
start = time()
b = sorted(a.keys(),key=lambda x:a[x],reverse=True)
print("took",str(time()-start))