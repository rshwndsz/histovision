import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None,
                 reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        if type(weight) != 'torch.Tensor' and weight is not None:
            weight = torch.tensor(weight)
        self.loss = nn.BCEWithLogitsLoss(weight=weight,
                                         size_average=size_average,
                                         reduce=reduce,
                                         reduction=reduction,
                                         pos_weight=pos_weight)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
