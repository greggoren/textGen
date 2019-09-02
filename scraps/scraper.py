import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from copy import deepcopy
from time import time

def add(a,b):
    return [i+j for i,j in zip(a,b)]

def div(a,d):
    return [i/d for i in a]

a = np.random.rand(10000)
b = np.random.rand(10000)
start = time()
avg = deepcopy(a)
print("numpy took:",start-time())

start = time()
avg = np.zeros(10000)
print("add took:",start-time())