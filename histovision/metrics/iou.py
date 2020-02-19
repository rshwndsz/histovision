import logging
import torch

logger = logging.getLogger('root')


def iou_score(preds, targets, smooth=1e-7):
    """Computes IoU or Jaccard index

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    smooth: float
        Smoothing for numerical stability
        1e-10 by default

    Returns
    -------
    iou : torch.Tensor
        IoU score or Jaccard index
    """
    intersection = torch.sum(targets * preds)
    union = torch.sum(targets) + torch.sum(preds) - intersection + smooth
    score = (intersection + smooth) / union

    return score
