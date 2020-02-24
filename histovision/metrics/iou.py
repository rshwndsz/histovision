import logging
import torch

logger = logging.getLogger('root')


def iou_score(y_pred, y_true, smooth=0.0, eps=1e-7, dims=None):
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
