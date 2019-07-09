import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from time import time

df = pd.DataFrame({'num_legs': [4, 2], 'num_wings': [0, 2]},index=['dog', 'hawk'])
print(df)
for row in df.itertuples():
    print(row[0])