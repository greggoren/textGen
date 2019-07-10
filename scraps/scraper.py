import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu
import random
import pandas as pd
from time import time




rows ={}
i=0
row={}
row["query"]="q"
row["input_sentence"] = "hi girl"
rows[0]=row
row={}
row["query"]="q"
row["input_sentence"] = "bye girl"
rows[1]=row
pd.DataFrame.from_dict(rows,orient="index").to_csv("test.csv")