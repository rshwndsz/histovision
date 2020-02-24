import logging
import torch

logger = logging.getLogger('root')


def true_positive(y_pred, y_true, num_classes=2):
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


def true_negative(y_pred, y_true, num_classes=2):
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


def false_positive(y_pred, y_true, num_classes=2):
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


def false_negative(y_pred, y_true, num_classes=2):
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


def precision_score(y_pred, y_true, num_classes=2):
    """Computes precision score

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
    precision : Tuple[torch.Tensor, ...]
        List of precision scores for each class
    """
    tp = true_positive(y_pred, y_true, num_classes).to(torch.float)
    fp = false_positive(y_pred, y_true, num_classes).to(torch.float)
    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def accuracy_score(y_pred, y_true, smooth=1e-10):
    """Compute accuracy score

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        Ground truths
    smooth: float
        Smoothing for numerical stability
        1e-10 by default

    Returns
    -------
    acc : torch.Tensor
        Average accuracy score
    """
    valids = (y_true >= 0)
    acc_sum = (valids * (y_pred == y_true)).sum().float()
    valid_sum = valids.sum().float()
    return acc_sum / (valid_sum + smooth)
