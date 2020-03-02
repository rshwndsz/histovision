import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_jaccard_score(probs, targets, smooth=0.0, eps=1e-7, dims=None):
    """Functional form

    Parameters
    ----------
    probs : torch.Tensor
        [N C *]
    targets : torch.Tensor
        [N C *]
    smooth : float
    eps : float
    dims : Tuple[int, ...]

    Returns
    -------
    scores : torch.Tensor
        [C]
    """
    assert probs.size() == targets.size()

    if dims is not None:
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
    else:
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union.clamp_min(eps) + smooth)

    return jaccard_score


def iou_score(outputs, targets, mode, include_classes=None, from_logits=True, smooth=0, eps=1e-7):
    assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
    if include_classes is not None:
        assert mode != BINARY_MODE, "Masking include_classes is not supported with mode=binary"
    assert outputs.size(0) == targets.size(0), "Batch size must be same"

    bs = targets.size(0)
    num_classes = outputs.size(1)
    dims = (0, 2)

    if from_logits:
        # Apply activations to get [0..1] class probabilities
        if mode == MULTICLASS_MODE:
            probs = outputs.softmax(dim=1)
        else:
            probs = outputs.sigmoid()
    else:
        probs = outputs

    if mode == BINARY_MODE:
        targets = targets.view(bs, 1, -1)                   # [N H W]   -> [N 1 *]
        probs = probs.view(bs, 1, -1)                       # [N 1 H W] -> [N 1 *]

    if mode == MULTICLASS_MODE:
        targets = targets.view(bs, -1)                      # [N H W]   -> [N *]
        probs = probs.view(bs, num_classes, -1)             # [N C H W] -> [N C *]
        targets = F.one_hot(targets.long(), num_classes)
        targets = targets.permute(0, 2, 1)                  # [N *]  -> [N C *]

    if mode == MULTILABEL_MODE:
        targets = targets.view(bs, num_classes, -1)
        probs = probs.view(bs, num_classes, -1)

    scores = soft_jaccard_score(probs, targets.type(probs.dtype),
                                smooth, eps, dims=dims)

    # Zero out contribution of channels that do not have true pixels
    mask = targets.sum(dims) > 0
    scores *= mask.float()

    # Include include_classes mentioned in `self.include_classes`
    if include_classes is not None:
        scores = scores[include_classes]

    return scores
