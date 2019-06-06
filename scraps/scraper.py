import numpy as np
# from torch.nn import LSTM
# import torch.nn.functional as F
import torch
x = torch.DoubleTensor([1,2,3])
rnn = torch.nn.LSTM(3,2,2)
h0 = torch.randn(3,1,1)
c0 = torch.randn(2, 3, 20)
print(h0)