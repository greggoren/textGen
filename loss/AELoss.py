# from torch.nn.modules.loss import _Loss
# import torch.nn.functional as F
import torch.nn.functional.softmax as softmax
# class AutoEncoderCrossEntropyLoss(_Loss):


    # def __init__(self, margin=1.0, size_average=True):
    #     super(AutoEncoderCrossEntropyLoss, self).__init__()
    #
    #
    # def forward(self, input1,input2, target):
    #     F.cross_entropy()
    #     return F.margin_ranking_loss(input1, input2,target, self.margin, self.size_average)


print(softmax([1,2,4],dim=3))