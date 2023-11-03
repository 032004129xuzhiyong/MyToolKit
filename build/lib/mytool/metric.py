# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
import keras.backend as K


def get_sparse_topk_metric(top_k):
    def top_metric(scores, labels, return_acc_list=False):
        """
        :param scores: [batch, num_classes]
        :param labels: [batch,]
        :param return_acc_list: whether to return origin accuracy list
        :return:
            acc: [] float topk metric
            accuracy list: [batch,] bool
        """
        accuracy_list = K.in_top_k(scores, labels, top_k).numpy()
        if return_acc_list:
            return accuracy_list.sum() / accuracy_list.size, accuracy_list
        return accuracy_list.sum() / accuracy_list.size
    setattr(top_metric,'__qualname__','Top'+str(top_k))
    return top_metric


def acc(scores, labels):
    return (scores.argmax(-1) == labels).float().mean()
