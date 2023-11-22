# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
import time
import torch
import os
import numpy as np
import pandas as pd
from mytool import tool
from mytool import plot
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
import functools
import random
import copy

def get_device():
    """
    :return:
        device: torch.device
    """
    detect_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return torch.device(detect_device)


def tensor_to_device(tensor_or_tensorlist, device):
    """
    map tensor or List[tensor]... to device
    :param tensor_or_tensorlist: tensor or List[tensor]
    :param device: torch device
    :return:
        tensor_or_tensorlist(device)
    """
    if torch.is_tensor(tensor_or_tensorlist):
        return tensor_or_tensorlist.to(device)
    else:
        return list(map(functools.partial(tensor_to_device, device=device), tensor_or_tensorlist))


def parser_compile_kw(compile_kw):
    """
    An auxiliary function
    :param compile_kw: Dict
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
    :return:
        loss_fn
        optimizer
        scheduler
        metric_list
    """
    loss_fn, optimizer = compile_kw['loss'], compile_kw['optimizer']
    scheduler = compile_kw['scheduler'] if 'scheduler' in compile_kw.keys() else None
    if 'metric' in compile_kw.keys():
        metric_list = compile_kw['metric'] if isinstance(compile_kw['metric'], (list, tuple)) else [
            compile_kw['metric']]
    else:  # None
        metric_list = []
    return loss_fn, optimizer, scheduler, metric_list


def get_optimizer_lr(optimizer):
    """get current optimizer lr"""
    for p in optimizer.param_groups:
        return p['lr']


def set_optimizer_lr(optimizer, cur_lr):
    """adjust torch.optim.optimizer lr rate"""
    for p in optimizer.param_groups:
        p['lr'] = cur_lr


def compute_loss(inputs, labels, model, loss_fn, device, loss_weights=None):
    """
    cope with a situation which may have more than one tensor(inputs,labels)/loss_fn
    :param inputs: tensor or List[tensor]
    :param labels: tensor or List[labels]
    :param model: torch model
    :param loss_fn: loss_fn or List[loss_fn]
    :param device: torch device
    :param loss_weights: None or List[numeric]
    :return:
        :key loss: for back propagation
        :key outputs: for metric_func
        :key labels: for metric_func
        :key loss_info_dict: Dict
            :key loss:
            :key loss_funcname_1/funcname_1_weight: maybe
            :key loss_funcname_2/funcname_2_weight: maybe
            ...
    """
    # init
    loss_info_dict = {}

    # get outputs and to list(has been to device)
    inputs = tensor_to_device(inputs, device)
    outputs = model(inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]  # rename

    # labels to list and to device
    if not isinstance(labels, (list, tuple)):  # tensor
        labels = [labels]
    labels = tensor_to_device(labels, device)
    assert len(outputs) == len(labels)

    # loss to list and has same dimension
    if not isinstance(loss_fn, (list, tuple)):  # func
        loss_list = [loss_fn for i in outputs]
    elif len(loss_fn) == 1:
        loss_list = [loss_fn[0] for i in outputs]
    elif len(loss_fn) == len(outputs):
        loss_list = loss_fn
    else:
        raise Exception('loss_fn must be correct for outputs!')
    loss_fn = loss_list  # rename
    assert len(loss_fn) == len(outputs)

    # set loss_weights
    if loss_weights is None:
        loss_weights = [1 for i in outputs]  # 同等权重
    assert len(loss_weights) == len(outputs)

    # normalize loss_weights
    lw_sum = np.sum(loss_weights)
    nlw = [torch.tensor(w / lw_sum, device=device) for w in loss_weights]
    loss_weights = nlw  # rename

    # Now outputs/labels/loss_fn/loss_weights are list
    # next step compute loss
    loss = torch.tensor(0.)
    for output_i, (out, lab, lf, lw) in enumerate(zip(outputs, labels, loss_fn, loss_weights)):
        # set func name
        lf_name = tool.get_func_name(lf)
        if len(outputs) > 1:
            lf_name += '_' + str(output_i + 1)

        # cal origin loss
        closs = lf(out, lab)

        # weight loss
        loss = loss + lw * closs

        # collect origin loss and loss weight
        if len(outputs) > 1:
            loss_info_dict['loss_' + lf_name] = closs.item()
            loss_info_dict[lf_name + '_weight'] = lw.item()

    # end for
    loss_info_dict['loss'] = loss.item()
    return loss, outputs, labels, loss_info_dict


def compute_single_metric(output, label, metric_func):
    """
    An auxiliary function for compute_metric
    :param output: tensor
    :param label: tensor
    :param metric_func: single_metric_func
    :return:
        :key metric_dict: Dict
            :key metric_name: metric_value
    """
    ndoutputs, ndlabels = output.cpu(), label.cpu()
    metric_value = metric_func(ndoutputs, ndlabels)
    metric_name = tool.get_func_name(metric_func)
    return {metric_name: metric_value}


def compute_metric(outputs, labels, metric_list):
    """
    cope with a situation which has more than one outputs/labels,
    and metric_list maybe have zero or one or many(more than len(outputs)) metric_func
    :param outputs: List[tensor] inherit from compute_loss
    :param labels: List[tensor] inherit from compute_loss
    :param metric_list: List[Union[metric_func,None]] or List[Union[List[metric_func], None]]
        shape:
            if it has two dimensions Or it has one dimension and len(metric_list) == len(outputs),
                suppose it corresponds to outputs/labels, len(metric_list) must be equal to len(outputs)
            else it must only have one dimension and it does not correspond to outputs/labels
        value:
            if element is None, it means that it does not need to calculate metric
    :return:
        metric_info_dict: Dict
            :key metric_metricname_1: value
            :key metric_metricname_2: value
    """
    metric_info_dict = {}
    if_two_dimension = False
    for el in metric_list:
        if isinstance(el, (tuple, list)):  # 只要有一个列表，就认为是二维
            if_two_dimension = True

    if_correspond = False
    if if_two_dimension or len(metric_list) == len(outputs):  # 认为是对应与outputs/labels
        assert len(metric_list) == len(outputs)
        if_correspond = True

    if if_correspond:  # 对应outputs/labels
        for output_ind, (mf_or_mf_list, output, label) in enumerate(zip(metric_list, outputs, labels)):
            if isinstance(mf_or_mf_list, (tuple, list)):  # list
                for mf in mf_or_mf_list:
                    if mf is not None:  # 是否计算loss
                        if len(outputs) == 1:  #
                            metric_info_dict.update(
                                tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                    'metric_'))
                        else:
                            metric_info_dict.update(
                                tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                    prefix='metric_',
                                                                    suffix='_' + str(output_ind + 1)))
            elif mf_or_mf_list is None:
                pass
            else:  # single func
                if len(outputs) == 1:
                    metric_info_dict.update(
                        tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf_or_mf_list),
                                                            'metric_'))
                else:
                    metric_info_dict.update(
                        tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf_or_mf_list),
                                                            prefix='metric_',
                                                            suffix='_' + str(output_ind + 1)))
    else:  # 一维 没有对应
        assert if_two_dimension == False
        # 可能只有一个metric_func,或者多于len(outputs)个metric_func
        for output_ind, (output, label) in enumerate(zip(outputs, labels)):
            for mf in metric_list:
                if mf is not None:  # 是否计算metric
                    if len(outputs) == 1:
                        metric_info_dict.update(
                            tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                'metric_'))
                    else:
                        metric_info_dict.update(
                            tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                prefix='metric_',
                                                                suffix='_' + str(output_ind + 1)))
    return metric_info_dict


def fit(model, dataload, epochs, compile_kw,
        device=None, val_dataload=None, loss_weights=None, callbacks=[],
        quiet=False):
    """
    similar to tf.keras.Model.fit
    :param model: torch model
    :param dataload: torch.utils.data.DataLoader for train
    :param epochs: iteration
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
    :param device: torch device
    :param val_dataload: torch.utils.data.DataLoader for validation
    :param loss_weights: List[numeric] or None calculate weight loss for every output
    :param callbacks: List[Callback] or CallbackList similar to tf.keras.callbacks
    :return:
        history_epoch:
            :key lr: 1-D epoch lr if it has a scheduler
            :key time/val_time: 1-D epoch_time
            :key loss/val_loss: 1-D epoch_loss
            :key loss_funcname_1/val_loss_funcname_1: 1-D one_output_loss
            :key funcname_1_weight/val_funcname_1_weight: 1-D one_output_loss_weight
            :key metric_metricName_1/val_metric_metricName_1: 1-D epoch_metric
            ...
    """
    history_epoch = History()
    # process args
    steps_per_epoch = len(dataload)
    validation_steps = None if val_dataload is None else len(val_dataload)
    device = device if device is not None else get_device()
    callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks,
                                                                                      model=model, params={
            'optimizer': compile_kw['optimizer']})
    callbacklist.append(PrintCallback(epochs, steps_per_epoch,quiet=quiet))

    # train begin
    callbacklist.on_train_begin(logs=None)
    for epoch_index in range(epochs):

        # epoch begin
        callbacklist.on_epoch_begin(epoch_index + 1, logs=None)

        # train one epoch
        epoch_logs = train_epoch(dataload, model, compile_kw, device, loss_weights, callbacks=callbacklist)

        # val one epoch
        if val_dataload is not None:
            val_epoch_logs = val_epoch(val_dataload, model, compile_kw, device, loss_weights)
        else:
            val_epoch_logs = {}

        # collect epoch logs
        history_epoch.update(epoch_logs)
        history_epoch.update(val_epoch_logs)

        # epoch end
        epoch_logs.update(val_epoch_logs)
        if callbacklist.on_epoch_end(epoch_index + 1, logs=epoch_logs): break

    # train end
    callbacklist.on_train_end(logs=None)

    return history_epoch


def train_epoch(dataload, model, compile_kw,
                device, loss_weights=None, callbacks=[]):
    """
    train dataload for one epoch
    :param dataload: torch.utils.data.DataLoader for train
    :param model: torch model
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
    :param device: torch device
    :param loss_weights: List[numeric] or None calculate weight loss for every output
    :param callbacks: List[Callback] or CallbackList similar to tf.keras.callbacks
    :return:
        epoch_logs:
            :key lr: float if it has a scheduler.
            :key time: float epoch time calculate sum batch time
            :key loss: float epoch loss calculate mean batch loss
            :key loss_funcname_1...: maybe
            :key funcname_1_weight...: maybe
            :key metric_metricName_1...: float epoch metric calculate mean batch metric
            ...
    """
    # epoch init
    callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list = parser_compile_kw(compile_kw)

    # get current epoch lr
    cur_epoch_lr = get_optimizer_lr(optimizer)

    model.train()
    # collect batch logs
    history_batch = History()
    for i, data in enumerate(dataload):
        # batch begin
        callbacklist.on_train_batch_begin(i + 1, logs=None)

        # train step
        batch_logs = train_step(data, compile_kw, model, device, loss_weights)

        # collect batch logs
        history_batch.update(batch_logs)

        # batch end
        callbacklist.on_train_batch_end(i + 1, logs=batch_logs)

    # end for
    epoch_logs = history_batch.mean(but_sum_for_keys=['time'])
    epoch_logs['lr'] = cur_epoch_lr
    # 调整学习率
    if scheduler is not None: scheduler.step()

    return epoch_logs


def val_epoch(dataload, model, compile_kw, device, loss_weights=None):
    """
    val or test dataload for one epoch
    :param dataload: torch.utils.data.DataLoader for validation
    :param model: torch model
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
    :param device: torch.device
    :param loss_weights: List[numeric] or None calculate weight loss for every output
    :return:
        val_epoch_logs:
            :key val_time: float epoch time calculate sum batch time
            :key val_loss: float epoch loss calculate mean batch loss
            :key val_loss_funcname_1...:
            :key val_funcname_1_weight...:
            :key val_metric_metricName_1: float epoch metric calculate mean batch metric
    """
    model.eval()
    # collect batch logs
    history_batch = History()
    with torch.no_grad():
        for i, vdata in enumerate(dataload):
            # test step
            batch_logs = test_step(vdata, compile_kw, model, device, loss_weights)

            # collect batch logs
            history_batch.update(batch_logs)

    # end for
    val_epoch_logs = history_batch.mean(but_sum_for_keys=['val_time'])

    return val_epoch_logs


def train_step(data, compile_kw, model, device, loss_weights=None):
    """
    Perform a single training step.

    Args:
        data (tuple): A tuple containing the inputs and labels for the batch.
        compile_kw (dict): A dictionary containing the compilation parameters for the model.
        model: The model to be trained.
        device: The device on which the model and data should be loaded.
        loss_weights (list, optional): A list of weights for each loss function. Defaults to None.

    Returns:
        dict: A dictionary containing the loss and metric information for the batch.
    """
    # batch init
    batch_logs = {}
    batch_start_time = time.time()

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list = parser_compile_kw(compile_kw)

    # forward and back propagation
    optimizer.zero_grad()
    inputs, labels = data
    # compute loss
    loss, outputs, labels, loss_info_dict = compute_loss(inputs, labels, model, loss_fn, device, loss_weights)
    loss.backward()
    optimizer.step()

    # collect loss
    batch_logs.update(loss_info_dict)

    # compute metric and collect metric
    metric_info_dict = compute_metric(outputs, labels, metric_list)
    batch_logs.update(metric_info_dict)

    # collect batch time
    batch_logs['time'] = time.time() - batch_start_time

    return batch_logs


def test_step(data, compile_kw, model, device, loss_weights=None):
    """
    Perform a single testing step.

    Args:
        data (tuple): A tuple containing the input data and labels.
        compile_kw (dict): A dictionary containing the compilation keywords.
        model: The model to be tested.
        device: The device on which the testing will be performed.
        loss_weights (list, optional): A list of loss weights. Defaults to None.

    Returns:
        dict: A dictionary containing the loss and metric information for the testing step.
    """
    # batch init
    batch_logs = {}
    batch_start_time = time.time()

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list = parser_compile_kw(compile_kw)

    # forward propagation
    vinputs, vlabels = data
    vloss, voutputs, vlabels, loss_info_dict = compute_loss(vinputs, vlabels, model, loss_fn, device,
                                                            loss_weights)

    # collect loss
    batch_logs.update(loss_info_dict)

    # compute metric and collect metric
    metric_info_dict = compute_metric(voutputs, vlabels, metric_list)
    batch_logs.update(metric_info_dict)

    # change name: add prefix 'val_' to key
    batch_logs = tool.add_pre_or_suf_on_key_for_dict(batch_logs, 'val_')

    # collect batch time
    batch_logs['val_time'] = time.time() - batch_start_time

    return batch_logs


class MyHyperParameter:
    """
    set hyperparameters for config
    if bool_new is True, every time it will return a new value from Int/Float/Boolean/Choice
    """
    def __init__(self, convergence_factor=1.0001):
        super().__init__()
        self.convergence_factor = convergence_factor
        self.last_int_dict = {}
        self.last_float_dict = {}
        self.last_bool_dict = {}
        self.last_choice_dict = {}
        self.bool_new = True
        self.all_name_weights = {}
        self.all_name_candi_list = {}

    def Int(self, name, min, max, step=1):
        candi_list = list(range(min,max+1,step))
        #收集用于调整各个候选参数的权重，初始化为1
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))#random.choice(candi_list)
            self.last_int_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_int_dict[name]

    def Float(self, name, min, max, step=1.):
        candi_list = list(np.arange(min,max,step))
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_float_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_float_dict[name]

    def Boolean(self, name):
        candi_list = [False, True]
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_bool_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_bool_dict[name]

    def Choice(self, name, candi_list):
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new:  # 第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_choice_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_choice_dict[name]

    def normalize(self, np_list):
        rowsum = np.sum(np_list)
        norm = np_list/rowsum
        return norm

    def update_configdict_weight(self, configdict):
        for name in configdict.keys():
            value = configdict[name]
            candi_list = self.all_name_candi_list[name]
            total_num = len(candi_list)
            #找到对应值的位置idx
            id = np.where(candi_list == value)
            id = int(id[0])
            #列出更新权重
            multi_w_leftlist =  np.linspace(1,self.convergence_factor,id,endpoint=False)
            multi_w_rightlist = np.linspace(self.convergence_factor,1,total_num-id)
            multi_w = np.concatenate([multi_w_leftlist,multi_w_rightlist],axis=0)
            self.all_name_weights[name] = self.all_name_weights[name] * multi_w

    def set_bool_new(self, bool_new):
        self.bool_new = bool_new

    def get_current_configdict(self):
        return_dict = {}
        return_dict.update(self.last_int_dict)
        return_dict.update(self.last_float_dict)
        return_dict.update(self.last_bool_dict)
        return_dict.update(self.last_choice_dict)
        return return_dict.copy()


class MyTuner:
    def __init__(self, max_trials, executions_per_trial=1, mode='min', quiet=False):
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.mode = mode
        self.hp = MyHyperParameter()
        self.configdict_list = []
        self.flag_list = []
        self.callbacklist = CallbackList([MyTunerPrintCallback(max_trials, executions_per_trial, quiet)])

    def run_trial(self, hp, **kwargs):
        #it will be rewritten.
        #运行一次实验，返回要检测的指标value
        pass

    def search(self, **kwargs):
        #运行所有trial，每个测验执行几次，统计平均值，收集指标和对应配置

        #tuner begin
        self.callbacklist.on_train_begin()

        #init
        tuner_start_time = time.time()

        for tri_idx in range(1, self.max_trials+1):

            #trial begin
            self.callbacklist.on_epoch_begin(tri_idx)

            #collect flag every trial
            per_trial_flags = []
            for ex_idx in range(1, self.executions_per_trial+1, 1):

                #execution begin
                self.callbacklist.on_train_batch_begin(ex_idx)

                #init execution
                batch_start_time = time.time()

                #same configdict for execution but new configdict for new trial.
                #for run_trial func
                if ex_idx == 1:
                    self.hp.set_bool_new(bool_new=True)
                else:
                    self.hp.set_bool_new(bool_new=False)

                #run once
                #it must be deepcopy, dict can not be modified in origin dict.
                copy_kwargs = copy.deepcopy(kwargs)
                flag = self.run_trial(self.hp, **copy_kwargs)

                # collect flag
                per_trial_flags.append(flag)

                #execution end
                self.callbacklist.on_train_batch_end(ex_idx, {'flag':flag,
                                                              'time':time.time()-batch_start_time})

            #collect configdict and mean flag
            current_trial_configdict = self.hp.get_current_configdict()
            current_trial_flag = float(np.mean(per_trial_flags))
            self.configdict_list.append(current_trial_configdict)
            self.flag_list.append(current_trial_flag)

            #sort for print
            self.sort_flag_and_configdict()

            #trial end 最佳配置、最佳flag、以及截止目前花费的tuner时间
            self.callbacklist.on_epoch_end(tri_idx, {'best_flag':self.flag_list[0],
                                                     'best_configdict':self.configdict_list[0],
                                                     'cur_flag':current_trial_flag,
                                                     'cur_configdict':current_trial_configdict,
                                                     'time': time.time()-tuner_start_time})

        #tuner end
        self.callbacklist.on_train_end()

    def sort_flag_and_configdict(self):
        # 按照mode排序，如果max，从大到小，如果min，从小到大
        self.flag_list = np.array(self.flag_list)
        if self.mode == 'min':
            argidx = np.argsort(self.flag_list)
        elif self.mode == 'max':
            argidx = np.argsort(-1 * self.flag_list)
        else:
            raise Exception('mode must be min or max')
        self.flag_list = list(self.flag_list[argidx])
        self.configdict_list = tool.config_dict_list_sort_with_argidx(self.configdict_list,argidx)
        #更新各个参数的权重
        self.hp.update_configdict_weight(self.configdict_list[0])

    def get_best_hyperparameters(self):
        #返回按最优到最差的配置列表
        return self.configdict_list

    def get_best_flags(self):
        return self.flag_list


class History:
    def __init__(self):
        self.history = {}  # all values must be 1-D List

    def update(self, other_dict):
        """
        append a number which has the same key
        :param other_dict:
        :return:
        """
        for key in other_dict.keys():
            if key in self.history.keys():
                self.history[key].append(other_dict[key])
            else:
                self.history[key] = [other_dict[key]]

    def clear(self):
        """reset dict"""
        self.history = {}

    def mean(self, but_sum_for_keys=[]):
        """
        calculate mean value for every key of dictionary
        :return:
            mean_dict
        """
        mean_dict = {}
        if isinstance(but_sum_for_keys, str): but_sum_for_keys = [but_sum_for_keys]
        for key in self.history.keys():
            if key in but_sum_for_keys:
                mean_dict[key] = np.sum(self.history[key])
            else:
                mean_dict[key] = np.mean(self.history[key])
        return mean_dict


class Callback:
    def __init__(self, params=None, model=None):
        self.params = params if params is not None else None
        self.model = model if model is not None else None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        pass

    def on_epoch_end(self, epoch, logs=None):  # only for train
        pass


class CallbackList(Callback):
    def __init__(self, callbacks, params=None, model=None):
        super().__init__(params, model)
        self.callbacks = list(callbacks)
        for cal in self.callbacks:
            if self.params is not None: cal.set_params(self.params)
            if self.model is not None: cal.set_model(self.model)

    def append(self, callback):
        if self.params is not None: callback.set_params(self.params)
        if self.model is not None: callback.set_model(self.model)
        self.callbacks.append(callback)

    def on_train_begin(self, logs=None):
        for cal in self.callbacks:
            cal.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cal in self.callbacks:
            cal.on_train_end(logs)

    def on_train_batch_begin(self, batch, logs=None):
        for cal in self.callbacks:
            cal.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for cal in self.callbacks:
            cal.on_train_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        for cal in self.callbacks:
            cal.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):  # only for train
        # special flag for early_stop_callback, if return True, stop training.
        break_flag = None
        for cal in self.callbacks:
            if cal.on_epoch_end(epoch, logs):
                break_flag = True
        return break_flag


class PrintCallback(Callback):
    """
    Print Training logs.
    """

    def __init__(self, total_epochs, steps_per_epoch,
                 batch_sep=100, if_micro_average=True, quiet=False, **kwargs):
        super().__init__(**kwargs)
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_sep = batch_sep
        self.if_micro_average = if_micro_average
        self.quiet = quiet
        self.micro_batch_history = History()

    def on_train_begin(self, logs=None):
        if not self.quiet:
            print('Begin Train!')

    def on_train_end(self, logs=None):
        if not self.quiet:
            print('End Train!')

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if self.if_micro_average:
            self.micro_batch_history.update(logs)

        if batch % self.batch_sep == 0:
            print_str = f'    [{batch}/{self.steps_per_epoch}] Batch {batch}'
            if self.if_micro_average:
                logs = self.micro_batch_history.mean()
                self.micro_batch_history.clear()
            for key in logs.keys():
                print_str += f' {key} {logs[key]}'
            if not self.quiet:
                print(print_str)
        else:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        if not self.quiet:
            print('=' * 20, f'[{epoch:04d}/{self.total_epochs:04d}]', ' Epoch ', f'{epoch} ', '=' * 20)

    def on_epoch_end(self, epoch, logs=None):
        print_str = f'Epoch {epoch}'
        for key in logs.keys():
            print_str += f' {key} {logs[key]}'
        if not self.quiet:
            print(print_str)


class DfSaveCallback(Callback):
    """
    Save epoch_logs to CSV
    """

    def __init__(self, df_save_path, **kwargs):
        super().__init__(**kwargs)
        self.df_save_path = df_save_path
        basedir = os.path.dirname(df_save_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.epoch_history = History()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)
        df = pd.DataFrame(self.epoch_history.history)
        df.to_csv(self.df_save_path, index=False, header=True)


class EarlyStoppingCallback(Callback):
    """
    如果有进步，那么patience_flag=patience，更新last_loss或last_mc(if restore_best_weights: save_model_weight)
    如果没有进步，那么patience_flag -=1，
    如果patience_flag==-1，return break_flag=True

    note:
        "进步": loss(down,mode=min), metric(up,mode=max)
        "dowm": cur_loss<last_loss and last_loss-cur_loss>=min_delta
                and baseline>cur_loss and baseline-cur_loss>=min_delta
        "up": cur_mc>last_mc and cur_mc-last_mc>=min_delta
                and cur_mc>baseline and cur_mc-baseline>=min_delta
    """

    def __init__(self, checkpoint_dir, monitor='val_loss', min_delta=0, patience=0,
                 mode='auto', baseline=None, restore_best_weights=False, save_best_only=False, quiet=False, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        if mode == 'auto':
            self.mode = 'min' if monitor.find('loss') != -1 else 'max'
        else:
            self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.save_best_only = save_best_only
        self.quit = quiet

        self.patience_flag = patience
        self.last_loss = float('inf') if self.mode == 'min' else None
        self.last_mc = 0. if self.mode == 'max' else None
        self.last_model_weight_path = None

    def save_weights(self):
        if_last_save = False
        if self.last_model_weight_path is not None:
            # 上次有保存过权重
            if_last_save = True
        # save
        new_model_weight_path = os.path.join(self.checkpoint_dir, tool.get_datetime_name() + '.pth')
        torch.save(self.model.state_dict(), new_model_weight_path)
        # 删除上个文件
        if if_last_save and self.save_best_only \
                and os.path.exists(self.last_model_weight_path):  # if remove last
            os.remove(self.last_model_weight_path)
        # 更新路径
        self.last_model_weight_path = new_model_weight_path

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            self.model.load_state_dict(torch.load(self.last_model_weight_path))

    def on_epoch_end(self, epoch, logs=None):
        break_flag = False
        if self.mode == 'min':  # loss
            cur_loss = logs[self.monitor]
            if self.baseline:  # 是否有baseline
                if cur_loss < self.last_loss and self.last_loss - cur_loss >= self.min_delta \
                        and self.baseline > cur_loss and self.baseline - cur_loss >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_loss = cur_loss
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
            else:
                if cur_loss < self.last_loss and self.last_loss - cur_loss >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_loss = cur_loss
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
        else:  # metric
            cur_mc = logs[self.monitor]
            if self.baseline:
                if cur_mc > self.last_mc and cur_mc - self.last_mc >= self.min_delta \
                        and cur_mc > self.baseline and cur_mc - self.baseline >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_mc = cur_mc
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
            else:
                if cur_mc > self.last_mc and cur_mc - self.last_mc >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_mc = cur_mc
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True

        if break_flag:
            if not self.quit:
                print('*' * 20, 'EarlyStopping!', '*' * 20)
            return True


class SchedulerWrapCallback(Callback):
    """
    custom scheduler_func has args: (cur_epoch,[cur_lr,[cur_epoch_logs,[util_now_epoch_history]]]).
    It update on epoch end and apply it lr_rate for next epoch.
    """

    def __init__(self, scheduler_func, manual=True, **kwargs):
        super().__init__(**kwargs)
        self.scheduler_func = scheduler_func
        self.manual = manual
        self.epoch_history = History()
        self.argcount = self.scheduler_func.__code__.co_argcount

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)
        cur_epoch, cur_lr, cur_epoch_logs = epoch, get_optimizer_lr(self.params['optimizer']), logs
        util_now_epoch_history = self.epoch_history
        if self.argcount == 1:
            new_lr = self.scheduler_func(cur_epoch)
        elif self.argcount == 2:
            new_lr = self.scheduler_func(cur_epoch, cur_lr)
        elif self.argcount == 3:
            new_lr = self.scheduler_func(cur_epoch, cur_lr, cur_epoch_logs)
        elif self.argcount == 4:
            new_lr = self.scheduler_func(cur_epoch, cur_lr, cur_epoch_logs, util_now_epoch_history)
        else:
            raise Exception('scheduler_func must have 1/2/3/4 parameters!!')
        if self.manual:
            return
        else:
            assert new_lr is not None
            set_optimizer_lr(self.params['optimizer'], new_lr)


class PlotLossMetricTimeLr(Callback):
    def __init__(self, loss_check='loss', metric_check='metric', time_check='time', lr_check='lr', **kwargs):
        super().__init__(**kwargs)
        self.loss_check = loss_check
        self.metric_check = metric_check
        self.time_check = time_check
        self.lr_check = lr_check
        self.epoch_history = History()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.epoch_history.history)
        fig = plot.plot_LossMetricTimeLr_with_df(df)
        plt.show()


class TunerRemovePreFileInDir(Callback):
    def __init__(self, dirs=None,  num_check_threshold=20, remove_ratio = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.dirs =[dirs]  if not isinstance(dirs,(tuple,list)) else dirs
        self.num_check_threshold = num_check_threshold
        self.remove_ratio = remove_ratio

    def on_train_end(self, logs=None):
        for dir in self.dirs:
            if dir is None:
                pass
            else:
                tool.remove_prefile_in_dir(dir,self.num_check_threshold,self.remove_ratio)
        return


class MyTunerPrintCallback(Callback):
    def __init__(self, max_trials, executions_per_trial, quiet=False):
        super().__init__()
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.quit = quiet

    def on_train_begin(self, logs=None):
        if not self.quit:
            print('Start Tuner!\n')

    def on_train_end(self, logs=None):
        if not self.quit:
            print('End Tuner!\n')

    def on_train_batch_begin(self, batch, logs=None):
        if not self.quit:
            print(f'Start Repeat #{batch}/{self.executions_per_trial}')

    def on_train_batch_end(self, batch, logs=None):
        #flag, time
        if not self.quit:
            print(f'End Repeat #{batch}/{self.executions_per_trial}',
                  '\tRepeat time: ', logs['time'],
                  '\tRepeat flag: ', logs['flag'])

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        if not self.quit:
            print(f'Start Trial #{epoch}/{self.max_trials}')

    def on_epoch_end(self, epoch, logs=None):  # only for train
        if not self.quit:
            print(f'End Trial #{epoch}/{self.max_trials}',
                  '\tUntil now total time: ', logs['time'],
                  '\tCurrent flag: ', logs['cur_flag'],
                  '\tBest flag: ', logs['best_flag'])

        cur_configdict = logs['cur_configdict']
        best_configdict = logs['best_configdict']
        tool.print_dicts_tablefmt([cur_configdict,best_configdict],['Current','Best'])


class WrapModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        module_list = nn.ModuleList([model])
        self.model = module_list[0]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        return self.model(inputs)

    def compile(self, loss, optimizer, metric=None, scheduler=None):
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        if metric is None:
            self.metric_list = []
        elif isinstance(metric, (tuple, list)):
            self.metric_list = metric
        else:
            self.metric_list = [metric]

    def fit(self, dataload,
            epochs,
            device=None,
            val_dataload=None,
            loss_weights=None,
            callbacks=[],
            quiet=False):
        # init
        self.epochs = epochs
        self.device = device if device is not None else get_device()
        self.loss_weights = loss_weights
        self.quiet = quiet
        steps_per_epoch = len(dataload)
        self.callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks,
                                            model=self, params={'optimizer': self.optimizer})
        self.callbacklist.append(PrintCallback(epochs, steps_per_epoch,quiet=quiet)) # default

        # train begin
        history_epoch = History()
        self.callbacklist.on_train_begin(logs=None)
        for epoch_index in range(epochs):

            # epoch begin
            self.callbacklist.on_epoch_begin(epoch_index + 1, logs=None)

            # train one epoch
            epoch_logs = self.train_epoch(dataload)

            # val one epoch
            if val_dataload is not None:
                val_epoch_logs = self.val_epoch(val_dataload)
            else:
                val_epoch_logs = {}

            # collect epoch logs
            history_epoch.update(epoch_logs)
            history_epoch.update(val_epoch_logs)

            # epoch end
            epoch_logs.update(val_epoch_logs)
            if self.callbacklist.on_epoch_end(epoch_index + 1, logs=epoch_logs): break

        # train end
        self.callbacklist.on_train_end(logs=None)

        return history_epoch

    def train_epoch(self, dataload):
        # get current epoch lr
        cur_epoch_lr = get_optimizer_lr(self.optimizer)

        self.train()
        history_batch = History()
        for i, data in enumerate(dataload):
            # batch begin
            self.callbacklist.on_train_batch_begin(i + 1, logs=None)

            # train step
            batch_logs = self.train_step(data)

            # collect batch logs
            history_batch.update(batch_logs)

            # batch end
            self.callbacklist.on_train_batch_end(i + 1, logs=batch_logs)

        # end for
        epoch_logs = history_batch.mean(but_sum_for_keys=['time'])
        epoch_logs['lr'] = cur_epoch_lr
        # 调整学习率
        if self.scheduler is not None: self.scheduler.step()

        return epoch_logs

    def val_epoch(self, val_dataload):
        self.eval()
        history_batch = History()
        with torch.no_grad():
            for i, vdata in enumerate(val_dataload):
                # test step
                batch_logs = self.test_step(vdata)

                # collect batch logs
                history_batch.update(batch_logs)

        # end for
        val_epoch_logs = history_batch.mean(but_sum_for_keys=['val_time'])

        return val_epoch_logs

    def train_step(self, data):
        # batch init
        batch_logs = {}
        batch_start_time = time.time()

        # forward and back propagation
        self.optimizer.zero_grad()
        inputs, labels = data
        # compute loss
        loss, outputs, labels, loss_info_dict = compute_loss(inputs, labels,
                                                             self, self.loss_fn,
                                                             self.device, self.loss_weights)
        loss.backward()
        self.optimizer.step()

        # collect loss
        batch_logs.update(loss_info_dict)

        # compute metric and collect metric
        metric_info_dict = compute_metric(outputs, labels, self.metric_list)
        batch_logs.update(metric_info_dict)

        # collect batch time
        batch_logs['time'] = time.time() - batch_start_time

        return batch_logs

    def test_step(self, data):
        # batch init
        batch_logs = {}
        batch_start_time = time.time()

        # forward propagation and compute loss
        vinputs, vlabels = data
        vloss, voutputs, vlabels, loss_info_dict = compute_loss(vinputs, vlabels,
                                                                self, self.loss_fn,
                                                                self.device, self.loss_weights)
        # collect loss
        batch_logs.update(loss_info_dict)

        # compute metric and collect metric
        metric_info_dict = compute_metric(voutputs, vlabels, self.metric_list)
        batch_logs.update(metric_info_dict)

        # collect batch time
        batch_logs['time'] = time.time() - batch_start_time

        # change name: add prefix 'val_' to key
        batch_logs = tool.add_pre_or_suf_on_key_for_dict(batch_logs, 'val_')

        return batch_logs





if __name__ == '__main__':
    class mydataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return (torch.randn(15, 3), torch.randn(3, 3)), (torch.randn(30, 6), torch.randn(6, 3))


    dataset = mydataset()
    dataload = torch.utils.data.DataLoader(dataset, batch_size=32)
    for x, l in dataload:
        print(type(x))
        break