import logging
import torch
from histovision.shared import utils

logger = logging.getLogger('root')


def dice_score(probs, targets, threshold: float = 0.5):
    """Calculate Sorenson-Dice coefficient

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities
    targets : torch.Tensor
        Ground truths
    threshold : float
        probs > threshold => 1
        probs <= threshold => 0

    Returns
    -------
    dice : torch.Tensor
        Dice score

    See Also
    --------
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """
    batch_size = targets.shape[0]
    with torch.no_grad():
        # Shape: [N, C, H, W]targets
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        # Shape: [N, C*H*W]
        if not (probs.shape == targets.shape):
            raise ValueError(f"Shape of probs: {probs.shape} must be the same"
                             f"as that of targets: {targets.shape}.")
        # Only 1's and 0's in p & t
        p = utils.predict(probs, threshold)
        t = utils.predict(targets, 0.5)
        # Shape: [N, 1]
        dice = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

    return utils.nanmean(dice)
