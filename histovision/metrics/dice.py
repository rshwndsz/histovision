import logging
import torch
from histovision.shared import utils

logger = logging.getLogger('root')


def dice_score(y_pred, y_true, smooth=0, eps=1e-7, dims=None):
    """Functional dice score

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    smooth : float
        Value to avoid divide by zero error
    eps : float
        Min value for cardinality
    dims : Tuple[int, ...]
        Sum along these dimensions

    Returns
    -------
    dice_score: torch.Tensor
        Dice score NOT loss
    """
    # Validate arguments
    if y_pred.size() != y_true.size():
        raise ValueError(f"size of predictions {y_pred.size()} != size of targets {y_true.size()} ")

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    score = (2.0 * intersection + smooth) / (cardinality.clamp_min(eps) + smooth)

    return score
