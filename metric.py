# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""


def acc(scores, labels):
    return (scores.argmax(-1) == labels).float().mean()
