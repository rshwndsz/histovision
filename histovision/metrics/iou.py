import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_jaccard_score(y_pred, y_true, smooth=0.0, eps=1e-7, dims=None):
    """Functional form

    Parameters
    ----------
    y_pred : torch.Tensor
        [N C *]
    y_true : torch.Tensor
        [N C *]
    smooth : float
    eps : float
    dims : Tuple[int, ...]

    Returns
    -------
    scores : torch.Tensor
        [C]
    """
    assert y_pred.size() == y_true.size()

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union.clamp_min(eps) + smooth)

    return jaccard_score


class IoUMetric(nn.Module):
    def __init__(self, mode, classes=None, from_logits=True, smooth=0, eps=1e-7):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(IoUMetric, self).__init__()
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
        y_true : torch.Tensor
            [N H W]

        Returns
        -------
        scores : torch.Tensor
            [C]
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

            y_true = F.one_hot(y_true.long(), num_classes)     # [N HW]   -> [N HW C]
            y_true = y_true.permute(0, 2, 1)                   # [N HW C] -> [N C HW]

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), self.smooth, self.eps, dims=dims)

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels

        mask = y_true.sum(dims) > 0
        scores *= mask.float()

        if self.classes is not None:
            scores = scores[self.classes]

        return scores


iou_score = IoUMetric(mode="multiclass")
