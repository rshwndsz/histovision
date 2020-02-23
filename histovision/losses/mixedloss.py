from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F

from .dice import DiceLoss
from .focal import FocalLoss


# Custom loss function combining Focal loss and Dice loss
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma)
        self.dice_loss = DiceLoss(mode="binary", log_loss=True)

    def forward(self, logits, target):
        loss = (self.alpha * self.focal_loss(logits, target) +
                self.dice_loss(logits, target))

        return loss.mean()
