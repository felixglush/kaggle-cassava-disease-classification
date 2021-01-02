from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


"""
Ideas:
- https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
- https://github.com/mlpanda/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss.py
- https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
"""

# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class SmoothedCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class BiTemperedLoss():
    def __init__(self):
        pass
    
    def forward(self, inputs, targets):
        pass