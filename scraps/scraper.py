import torch

src = torch.FloatTensor([[[2, 1], [2, 2]], [[2, 1], [2, 2]]])
tgt = torch.LongTensor([[0,1],[1,1]])
src =[torch.FloatTensor([[2, 1], [2, 2]]).reshape((2, 2, 1)), torch.FloatTensor([[2, 1], [2, 2]]).reshape((2, 2, 1))]
src = torch.cat(src, dim=2)
c = torch.nn.CrossEntropyLoss(reduction='none')
loss = c(src,tgt)

print(loss)


