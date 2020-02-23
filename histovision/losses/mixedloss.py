from torch import nn
from .dice import DiceLoss
from .focal import FocalLoss


# Custom loss function combining Focal loss and Dice loss
class MixedLoss(nn.Module):
    def __init__(self, gamma, loss_weights=(1, 1)):
        super().__init__()
        self.lw = loss_weights
        self.focal_loss = FocalLoss(gamma)
        self.dice_loss = DiceLoss(mode="binary", log_loss=True)

    def forward(self, logits, target):
        loss = (self.lw[0] * self.focal_loss(logits, target) +
                self.lw[1] * self.dice_loss(logits, target))

        return loss.mean()
