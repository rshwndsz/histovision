import logging
import torch

__all__ = ["precision", "accuracy", "recall", "f1"]

logger = logging.getLogger('root')

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def _true_positive(y_pred, y_true, num_classes=2):
    """Compute number of true positive predictions

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tp : torch.Tensor
        Tensor of number of true positives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((y_pred == i) & (y_true == i)).sum())

    return torch.tensor(out)


def _true_negative(y_pred, y_true, num_classes=2):
    """Computes number of true negative predictions

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tn : torch.Tensor
        Tensor of true negatives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((y_pred != i) & (y_true != i)).sum())

    return torch.tensor(out)


def _false_positive(y_pred, y_true, num_classes=2):
    """Computes number of false positive predictions

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fp : torch.Tensor
        Tensor of false positives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((y_pred == i) & (y_true != i)).sum())

    return torch.tensor(out)


def _false_negative(y_pred, y_true, num_classes=2):
    """Computes number of false negative predictions

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fn : torch.Tensor
        Tensor of false negatives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((y_pred != i) & (y_true == i)).sum())

    return torch.tensor(out)


def precision(outputs, targets, mode, from_logits, include_classes=None):
    """Precision

    Parameters
    ----------
    outputs : torch.Tensor
    targets : torch.Tensor
    from_logits : bool
    mode : str
    include_classes : Tuple[int, ...]

    Returns
    -------
    torch.Tensor
    """
    assert outputs.size(0) == targets.size(0), "Batch size must be same"
    num_classes = outputs.size(1)
    probs = outputs
    if from_logits:
        if mode == MULTICLASS_MODE:
            probs = outputs.softmax(dim=1)
        else:
            probs = outputs.sigmoid()
    preds = probs.argmax(dim=1)

    tp = _true_positive(preds, targets, num_classes)
    fp = _false_positive(preds, targets, num_classes)

    scores = tp.float() / (tp.float() + fp.float())
    scores[torch.isnan(scores)] = 0

    if include_classes is not None:
        scores = scores[include_classes]
    return scores


def accuracy(outputs, targets, mode, from_logits, include_classes=None):
    """Accuracy

    Parameters
    ----------
    outputs : torch.Tensor
    targets : torch.Tensor
    mode : str
    from_logits : bool
    include_classes : Tuple[int, ...]

    Returns
    -------
    torch.Tensor
    """
    assert outputs.size(0) == targets.size(0), "Batch size must be same"
    num_classes = outputs.size(1)
    probs = outputs
    if from_logits:
        if mode == MULTICLASS_MODE:
            probs = outputs.softmax(dim=1)
        else:
            probs = outputs.sigmoid()
    preds = probs.argmax(dim=1)

    tp = _true_positive(preds, targets, num_classes)
    tn = _true_negative(preds, targets, num_classes)
    total_population = targets.view(-1).size(0)

    scores = (tp.float() + tn.float()) / float(total_population)
    scores[torch.isnan(scores)] = 0

    if include_classes is not None:
        scores = scores[include_classes]
    return scores


def recall(outputs, targets, mode, from_logits, include_classes=None):
    """Recall

    Parameters
    ----------
    outputs : torch.Tensor
    targets : torch.Tensor
    mode : str
    from_logits : bool
    include_classes : Tuple[int, ...]

    Returns
    -------
    torch.Tensor
    """
    assert outputs.size(0) == targets.size(0), "Batch size must be same"
    num_classes = outputs.size(1)
    probs = outputs
    if from_logits:
        if mode == MULTICLASS_MODE:
            probs = outputs.softmax(dim=1)
        else:
            probs = outputs.sigmoid()
    preds = probs.argmax(dim=1)

    tp = _true_positive(preds, targets, num_classes)
    fn = _false_negative(preds, targets, num_classes)

    scores = (tp.float()) / (tp.float() + fn.float())
    scores[torch.isnan(scores)] = 0

    if include_classes is not None:
        scores = scores[include_classes]
    return scores


def f1(outputs, targets, mode, from_logits, include_classes=None):
    """F1

    Parameters
    ----------
    outputs : torch.Tensor
    targets : torch.Tensor
    mode : str
    from_logits : bool
    include_classes : Tuple[int, ...]

    Returns
    -------
    torch.Tensor
    """
    p = precision(outputs, targets,
                  mode=mode, from_logits=from_logits, include_classes=None)
    r = recall(outputs, targets,
               mode=mode, from_logits=from_logits, include_classes=None)

    scores = 2 * (p.float() * r.float()) / (p.float() + r.float())
    scores[torch.isnan(scores)] = 0

    if include_classes is not None:
        scores = scores[include_classes]
    return scores
