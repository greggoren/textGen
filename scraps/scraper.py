import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from copy import deepcopy
from time import time
import pickle

def div(a,d):
    return [i/d for i in a]

a = np.random.rand(300)
b = np.random.rand(10000)
with open("test.pkl",'wb') as t:
    pickle.dump(a,t)

start = time()
with open("test.pkl",'rb') as t:
    c = pickle.load(t)
print("took:",start-time())

