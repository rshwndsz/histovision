# Credits: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/jaccard.py
import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_jacard_score(y_pred, y_true, smooth=0.0, eps=1e-7, dims=None):
    """Jaccard score

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    smooth : float
        Constant for numerical stability
    eps : float
        Constant for numerical stability
    dims : Tuple[int, ...]
        Dimensions to sum over

    Returns
    -------
    jaccard_score: torch.Tensor
        Jaccard score for each class
    """
    assert y_pred.size() == y_true.size()

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)

    union = cardinality - intersection
    score = (intersection + smooth) / (union.clamp_min(eps) + smooth)

    return score


class JaccardLoss(nn.Module):
    """Jaccard loss for semgmentation & classification"""

    def __init__(self, mode, include_classes=None, log_loss=False, from_logits=True, smooth=0, eps=1e-7):
        """Initialize

        Parameters
        ----------
        mode :
            Metric mode {'binary', 'multiclass', 'multilabel'}
        include_classes :
            Optional list of classes that contribute in loss computation;
            By default, all channels are included.
        log_loss :
            If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        from_logits :
            If True assumes input is raw outputs
        smooth :
        eps :
            Small epsilon for numerical stability
        """
        super(JaccardLoss, self).__init__()
        # Validate arguments
        if mode not in [BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE]:
            raise ValueError(f"Mode must be one of [binary, multiclass, multilabel]")
        if include_classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            include_classes = torch.tensor(include_classes, dtype=torch.long)

        self.mode = mode
        self.include_classes = include_classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred, y_true):
        """Forward pass

        Parameters
        ----------
        y_pred :
            Predictions
        y_true :
            Ground truths

        Returns
        -------
        dice_loss : torch.Tensor
            Dice Loss NOT Score
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                # dim=1 => every slice along dimension 'C' sums to 1
                y_pred = y_pred.softmax(dim=1)
            else:
                # All values range from [0..1]
                y_pred = y_pred.sigmoid()

        bs = y_true.size(0)             # N
        num_classes = y_pred.size(1)    # C
        dims = (0, 2)                   # (N, -1) from N C HW

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)             # N C=1 HW
            y_pred = y_pred.view(bs, 1, -1)             # N C=1 HW

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)                # N HW   [0..C-1]
            y_pred = y_pred.view(bs, num_classes, -1)   # N C HW [0..1]

            y_true = F.one_hot(y_true, num_classes)     # N HW C [0..1]
            y_true = y_true.permute(0, 2, 1)            # N C HW [0..1]

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)   # N C HW
            y_pred = y_pred.view(bs, num_classes, -1)   # N C HW

        scores = soft_jacard_score(y_pred, y_true.type_as(y_pred), self.smooth, self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = torch.tensor(1) - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.include_classes is not None:
            loss = loss[self.include_classes]

        return loss.mean()
