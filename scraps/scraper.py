import torch
import numpy as np
# src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])

src = torch.FloatTensor([[2,1],[2,2],[0,1]])
tgt = torch.LongTensor([0,1,100])
c = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=100)
loss = c(src,tgt)
print(loss)

l = torch.LongTensor([3,2,4])
# l = torch.FloatTensor(l)
for i in range(loss.shape[0]):
    loss[i]=loss[i]/float(l[i])
print(loss)



