import logging
import torch
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger('root')


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_dice_score(y_pred, y_true, smooth=0, eps=1e-7, dims=None):
    """Functional form

    Parameters
    ----------
    y_pred : torch.Tensor
        [N C HW]
    y_true : torch.Tensor
        [N C HW]
    smooth : float
    eps : float
    dims : Tuple[int, ...]

    Returns
    -------
    dice_scores : torch.Tensor
    """
    assert y_pred.size() == y_true.size()
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
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

    def forward(self, y_pred, y_true):
        """Forward pass

        Parameters
        ----------
        y_pred : torch.Tensor
            [N C H W]
        y_true :
            [N H W]

        Returns
        -------
        dice_scores : torch.Tensor
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.softmax(dim=1)
            else:
                y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true.long(), num_classes)      # [N HW]   -> [N HW C]
            y_true = y_true.permute(0, 2, 1)                    # [N HW C] -> [N C HW]

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), self.smooth, self.eps, dims=dims)

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        mask = y_true.sum(dims) > 0
        scores *= mask.float()

        if self.classes is not None:
            scores = scores[self.classes]

        return scores


dice_score = DiceMetric(mode="multiclass")
