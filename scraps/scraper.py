import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random

print(random.random())
a = torch.FloatTensor([[1,1],[2,2]])
softmax = torch.nn.Softmax(dim=1)
print(softmax(a))