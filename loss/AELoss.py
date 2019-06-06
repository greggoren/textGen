from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import numpy as np
class AutoEncoderCrossEntropyLoss(_Loss):


    def __init__(self):
        super(AutoEncoderCrossEntropyLoss, self).__init__()


    def forward(self, decoded_output,input2, target):
        F.cross_entropy()
        return F.margin_ranking_loss(input1, input2,target, self.margin, self.size_average)


