import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        if type(weight) != 'torch.Tensor' and weight is not None:
            weight = torch.tensor(weight)
        self.loss = nn.CrossEntropyLoss(weight=weight,
                                        size_average=size_average,
                                        ignore_index=ignore_index,
                                        reduce=reduce,
                                        reduction=reduction)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
