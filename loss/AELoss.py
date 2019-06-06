# from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import numpy as np
# from torch.nn.functional import softmax
# class AutoEncoderCrossEntropyLoss(_Loss):


    # def __init__(self, margin=1.0, size_average=True):
    #     super(AutoEncoderCrossEntropyLoss, self).__init__()
    #
    #
    # def forward(self, input1,input2, target):
    #     F.cross_entropy()
    #     return F.margin_ranking_loss(input1, input2,target, self.margin, self.size_average)


print(F.softmax(torch.DoubleTensor([1,2,4])))