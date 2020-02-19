# Imports
# PyTorch
import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F


class DiceLoss(nn.Module):
    # See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def forward(self, logits, target, smooth=1e-7):
        probs = torch.sigmoid(logits)
        iflat = probs.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    # See: https://arxiv.org/abs/1708.02002
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        if not (target.size() == logits.size()):
            raise ValueError(f"Target size ({target.size()}) must be the same "
                             f"as input size ({logits.size()})")
        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()
        invprobs = F.logsigmoid(-logits * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


# Custom loss function combining Focal loss and Dice loss
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma)
        self.dice_loss = DiceLoss()

    def forward(self, logits, target):
        loss = (self.alpha * self.focal_loss(logits, target) -
                torch.log(self.dice_loss(logits, target)))

        return loss.mean()
