import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
a = pd.DataFrame(columns=["test"])
a["test"]={0:{1:0,2:1},1:{1:0,2:1}}
print(a)