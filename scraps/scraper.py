import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])



l = torch.LongTensor([3,2,4])
a,b = torch.max(l,dim=0)
print(a,b.item())


