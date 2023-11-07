# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
from sklearn import metrics
import torch


###################################################################
###################################################################
#                         分类
###################################################################
###################################################################


def acc(scores, labels):
    """
    Accuracy
    Args:
        scores: 2D torch.tensor
        labels: 1D

    Returns:
        acc
    """
    return (scores.argmax(-1) == labels).float().mean()


def precision(scores, labels, binary=False):
    """
    Precision
    Args:
        scores: 2D torch.tensor
        labels: 1D
        binary: bool

    Returns:
        precision
    """
    if binary:
        s = metrics.precision_score(labels, scores.argmax(-1), average='binary')
    else:
        s = metrics.precision_score(labels, scores.argmax(-1), average='micro')
    return s


def recall(scores, labels, binary=False):
    """
    recall
    Args:
        scores: 2D torch.tensor
        labels: 1D
        binary:

    Returns:
        recall
    """
    if binary:
        s = metrics.recall_score(labels, scores.argmax(-1), average='binary')
    else:
        s = metrics.recall_score(labels, scores.argmax(-1), average='micro')
    return s


def f1(scores, labels, binary=False):
    """
    f1_score
    Args:
        scores: 2D torch.tensor
        labels: 1D
        binary:

    Returns:
        f1_score
    """
    if binary:
        s = metrics.f1_score(labels, scores.argmax(-1), average='binary')
    else:
        s = metrics.f1_score(labels, scores.argmax(-1), average='micro')
    return s


def get_topk(topk, return_correct_num=False):
    """
    get topk function
    Args:
        topk: int >=1
        return_correct_num: bool whether to return correct number from topk_func

    Returns:
        topk_func
    """

    def topk_metric(scores, labels):
        """
        topk metric
        Args:
            scores: 2D torch.tensor
            labels: 1D

        Returns:
            topk_metric
        """
        if return_correct_num:
            num = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=False, labels=range(scores.shape[-1]))
            return num
        else:
            s = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=True, labels=range(scores.shape[-1]))
            return s

    setattr(topk_metric, '__qualname__', 'top' + str(topk))
    return topk_metric
