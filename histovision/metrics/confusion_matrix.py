import logging
import torch

logger = logging.getLogger('root')


def true_positive(preds, targets, num_classes=2):
    """Compute number of true positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
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
        out.append(((preds == i) & (targets == i)).sum())

    return torch.tensor(out)


def true_negative(preds, targets, num_classes=2):
    """Computes number of true negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
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
        out.append(((preds != i) & (targets != i)).sum())

    return torch.tensor(out)


def false_positive(preds, targets, num_classes=2):
    """Computes number of false positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
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
        out.append(((preds == i) & (targets != i)).sum())

    return torch.tensor(out)


def false_negative(preds, targets, num_classes=2):
    """Computes number of false negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
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
        out.append(((preds != i) & (targets == i)).sum())

    return torch.tensor(out)


def precision_score(preds, targets, num_classes=2):
    """Computes precision score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    precision : Tuple[torch.Tensor, ...]
        List of precision scores for each class
    """
    tp = true_positive(preds, targets, num_classes).to(torch.float)
    fp = false_positive(preds, targets, num_classes).to(torch.float)
    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def accuracy_score(preds, targets, smooth=1e-10):
    """Compute accuracy score

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
    acc : torch.Tensor
        Average accuracy score
    """
    valids = (targets >= 0)
    acc_sum = (valids * (preds == targets)).sum().float()
    valid_sum = valids.sum().float()
    return acc_sum / (valid_sum + smooth)
