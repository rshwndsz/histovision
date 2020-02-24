from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F


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
