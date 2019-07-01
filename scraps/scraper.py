import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random

a = torch.FloatTensor([[1,1,4,2,11],[2,2,2,2,2]])
lens = torch.LongTensor([2,3])
mask = torch.arange(a.shape[1])[None, :] <lens[:, None]
a[~mask]=-float('inf')
print(a)
