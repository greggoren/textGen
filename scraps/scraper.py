import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
from  nltk.translate.bleu_score import sentence_bleu


ref = [['he', 'founded', '<EOS>']]
cand = ['<EOS>']
print(sentence_bleu(ref,cand,weights=(1,)))