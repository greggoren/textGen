import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])


a = torch.LongTensor([[[1]*5,[2]*5],[[1]*5,[2]*5]])

print(a.reshape(1,a.shape[0],2*a[0].shape[1]))
