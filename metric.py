    # -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
from sklearn import metrics
import torch
import torch.nn.functional as F


###################################################################
###################################################################
#                         分类
###################################################################
###################################################################


def acc(scores, labels):
    """
    Computes the accuracy of a model's predictions.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        float: The accuracy of the model.
    """
    return (scores.argmax(-1) == labels).float().mean()


def precision(scores, labels, binary=False):
    """
    Calculates the precision score for a given set of predictions and labels.

    Args:
        scores: 2D torch.tensor containing the predicted scores for each class.
        labels: 1D tensor containing the true labels for each prediction.
        binary: bool indicating whether to calculate precision for binary classification or not.

    Returns:
        precision score as a float.
    """
    if binary:
        s = metrics.precision_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.precision_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def recall(scores, labels, binary=False):
    """
    Calculate recall score.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.
        binary (bool, optional): If True, calculate binary recall. Default is False.

    Returns:
        float: The recall score.
    """
    if binary:
        s = metrics.recall_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.recall_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def f1(scores, labels, binary=False):
    """
    Calculate the F1 score.

    Args:
        scores (torch.Tensor): 2D tensor of predicted scores.
        labels (torch.Tensor): 1D tensor of true labels.
        binary (bool): Whether to calculate the F1 score for binary classification or not.

    Returns:
        float: The F1 score.
    """
    if binary:
        s = metrics.f1_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.f1_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def get_topk(topk, return_correct_num=False):
    """
    Returns a function that calculates the top-k accuracy metric for a given k.

    Args:
        topk (int): The value of k for which the top-k accuracy metric is to be calculated. Must be >= 1.
        return_correct_num (bool): Whether to return the number of correct predictions instead of the accuracy score.

    Returns:
        A function that calculates the top-k accuracy metric for a given set of scores and labels.
    """
    def topk_metric(scores, labels):
        """
        Computes the top-k accuracy score for a given set of predicted scores and true labels.

        Args:
            scores (torch.Tensor): A 2D tensor of predicted scores.
            labels (torch.Tensor): A 1D tensor of true labels.

        Returns:
            The top-k accuracy score.
        """
        if return_correct_num:
            num = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=False, labels=range(scores.shape[-1]))
            return num
        else:
            s = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=True, labels=range(scores.shape[-1]))
            return s

    setattr(topk_metric, '__qualname__', 'top' + str(topk))
    return topk_metric



def auc_metric(scores, labels):
    """
    Computes the area under the ROC curve (AUC) score for a given set of predicted scores and true labels.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        The AUC score.
    """
    scores_softmax = F.softmax(scores, dim=1)
    return metrics.roc_auc_score(labels, scores_softmax, multi_class='ovr')


if __name__ == '__main__':
    pass