import logging
import torch
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger('root')


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_dice_score(probs, targets, smooth=0, eps=1e-7, dims=None):
    """Functional form

    Parameters
    ----------
    probs : torch.Tensor
        [N C HW]
    targets : torch.Tensor
        [N C HW]
    smooth : float
    eps : float
    dims : Tuple[int, ...]

    Returns
    -------
    dice_scores : torch.Tensor
    """
    assert probs.size() == targets.size()
    if dims is not None:
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
    else:
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
    scores = (2.0 * intersection + smooth) / (cardinality.clamp_min(eps) + smooth)

    return scores


class DiceMetric(nn.Module):
    def __init__(self, mode, classes=None, from_logits=True, smooth=0, eps=1e-7):
        super(DiceMetric, self).__init__()
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, outputs, targets):
        """Forward pass

        Parameters
        ----------
        outputs : torch.Tensor  [N C H W]
            Outputs of the model
            May be raw logits based on `self.from_logits`
        targets : torch.Tensor  [N H W]
            Ground truths
            Have values in {0, C-1}
        Returns
        -------
        dice_scores : torch.Tensor
        """
        assert outputs.size(0) == targets.size(0), "Batch size must be same"

        bs = targets.size(0)
        num_classes = outputs.size(1)
        dims = (0, 2)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                probs = outputs.softmax(dim=1)
            else:
                probs = outputs.sigmoid()
        else:
            probs = outputs

        if self.mode == BINARY_MODE:
            targets = targets.view(bs, 1, -1)           # [N H W]   -> [N 1 *]
            probs = probs.view(bs, 1, -1)               # [N 1 H W] -> [N 1 *]

        if self.mode == MULTICLASS_MODE:
            targets = targets.view(bs, -1)              # [N H W]   -> [N *]
            probs = probs.view(bs, num_classes, -1)     # [N C H W] -> [N C *]
            targets = F.one_hot(targets.long(), num_classes)
            targets = targets.permute(0, 2, 1)          # [N *]  -> [N C *]

        if self.mode == MULTILABEL_MODE:
            targets = targets.view(bs, num_classes, -1)
            probs = probs.view(bs, num_classes, -1)

        scores = soft_dice_score(probs, targets.type_as(probs),
                                 self.smooth, self.eps, dims=dims)

        # zero contribution of channels that does not have true pixels
        mask = targets.sum(dims) > 0
        scores *= mask.float()

        # Include classes mentioned in `self.classes`
        if self.classes is not None:
            scores = scores[self.classes]

        return scores
