import torch

c = torch.nn.CrossEntropyLoss()
src = torch.FloatTensor([[[1,30],[1,2]]])
tgt = torch.LongTensor([[0,0]])

l= c(src,tgt)
print(l.item())