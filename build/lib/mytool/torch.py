# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
import time
import torch
import numpy as np
from mytool import tool

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


def train_one_epoch(dataload, model, epoch_index, tb_writer,
                    compile_kw, other_kw,
                    print_batch_sep = 500,
                    ):
    """
    train one epoch in pytorch
    :param dataload: torch.utils.data.DataLoader
    :param model: torch model
    :param epoch_index: int epoch index
    :param tb_writer: torch.utils.tensorboard.SummaryWriter
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
    :param other_kw:
        :key device[optional]
        :key train_num_steps_per_epoch: int
    :return:
        info: Dict
            :key epoch_loss float
            :key epoch_time float
            :key epoch_metric_dict Dict[str,float] dict[metric_name]=metric_value
    """
    info = {}
    epoch_start_time = time.time()
    #compile_kw
    loss_fn, optimizer = compile_kw['loss'], compile_kw['optimizer']
    scheduler = compile_kw['scheduler'] if 'scheduler' in compile_kw.keys() else None
    metric_list = compile_kw['metric'] if isinstance(compile_kw['metric'],(list,tuple)) else [compile_kw['metric']]
    metric_name_list = [tool.get_func_name(metric_func) for metric_func in metric_list]
    #other_kw
    device = other_kw['device'] if 'device' in other_kw.keys() else get_device()
    train_num_steps_per_epoch = other_kw['train_num_steps_per_epoch']

    model.train()
    info['epoch_loss'], info['batch_loss'] = 0., 0. #float, float
    #[float,] for per metric   float for per metric
    info['epoch_metric_list'], info['batch_metric_list'] = [[] for i in metric_list], [0. for i in metric_list]
    for i,data in enumerate(dataload):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()

        #compute and collect metric
        for midx, metric_func in enumerate(metric_list):
            ndoutputs, ndlabels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            nd_metric = metric_func(ndoutputs,ndlabels)
            info['batch_metric_list'][midx] += nd_metric
            info['epoch_metric_list'][midx].append(nd_metric)

        #collect loss
        info['batch_loss'] += loss.item()
        info['epoch_loss'] += loss.item()

        #print
        if i % print_batch_sep == print_batch_sep-1:
            tb_x = epoch_index * train_num_steps_per_epoch + i + 1
            print_str = f'  batch {i+1}'
            #write loss
            batch_loss = info['batch_loss'] / print_batch_sep
            tb_writer.add_scaler('Batch_Loss/train',batch_loss,tb_x)
            print_str += f' loss: {batch_loss}'
            #clear batch_loss
            info['batch_loss'] = 0.
            #write metric
            for midx in range(len(metric_list)):
                batch_metric = info['batch_metric_list'][midx] / print_batch_sep
                tb_writer.add_scaler('Batch_'+metric_name_list[midx]+'/train', batch_metric, tb_x)
                print_str += ' ' + metric_name_list[midx] + ': ' + f'{batch_metric}'
                #clear batch_metric_list
                info['batch_metric_list'][midx] = 0.
            print(print_str)

    #end for
    #epoch loss
    epoch_loss = info['epoch_loss'] / train_num_steps_per_epoch
    tb_writer.add_scalar('Epoch_Loss/train', epoch_loss, epoch_index)
    #epoch metric
    for midx in range(len(metric_list)):
        epoch_metric = np.mean(info['epoch_metric_list'][midx])
        tb_writer.add_scalar('Epoch_'+metric_name_list[midx]+'/train', epoch_metric, epoch_index)
        #change List item into float item
        info['epoch_metric_list'][midx] = epoch_metric
    #epoch time
    epoch_time = time.time() - epoch_start_time
    #make return info
    info['epoch_metric_dict'] = dict(zip(metric_name_list,info['epoch_metric_list']))
    info['epoch_loss'] = epoch_loss
    info['epoch_time'] = epoch_time
    info.pop('batch_loss')
    info.pop('batch_metric_list')
    return info

def val_one_epoch(dataload, model, epoch_index, tb_writer,
                    compile_kw, other_kw,
                    ):
    """
    train one epoch in pytorch
    :param dataload: torch.utils.data.DataLoader
    :param model: torch model
    :param epoch_index: int epoch index
    :param tb_writer: torch.utils.tensorboard.SummaryWriter
    :param compile_kw:
        :key loss: compute loss
        :key metric: func(outputs,labels) or List[func]
    :param other_kw:
        :key device[optional]
        :key val_num_steps_per_epoch: int
    :return:
        info: Dict
            :key epoch_loss float
            :key epoch_time float
            :key epoch_metric_dict Dict[str,float] dict[metric_name]=metric_value
    """
    info = {}
    epoch_start_time = time.time()
    #compile_kw
    loss_fn = compile_kw['loss']
    metric_list = compile_kw['metric'] if isinstance(compile_kw['metric'],(list,tuple)) else [compile_kw['metric']]
    metric_name_list = [tool.get_func_name(metric_func) for metric_func in metric_list]
    #other_kw
    device = other_kw['device'] if 'device' in other_kw.keys() else get_device()
    val_num_steps_per_epoch = other_kw['val_num_steps_per_epoch']

    model.eval()
    info['epoch_loss'] = 0.
    #[float,] for per metric
    info['epoch_metric_list'] = [[] for i in metric_list]
    with torch.no_grad():
        for i, vdata in enumerate(dataload):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            # collect loss
            info['epoch_loss'] += vloss.item()
            # compute and collect metric
            for midx, metric_func in enumerate(metric_list):
                ndoutputs, ndlabels = voutputs.cpu().detach().numpy(), vlabels.cpu().detach().numpy()
                nd_metric = metric_func(ndoutputs, ndlabels)
                info['epoch_metric_list'][midx].append(nd_metric)
    #end for
    #epoch loss
    epoch_loss = info['epoch_loss'] / val_num_steps_per_epoch
    tb_writer.add_scalar('Epoch_Loss/val', epoch_loss, epoch_index)
    #epoch metric
    for midx in range(len(metric_list)):
        epoch_metric = np.mean(info['epoch_metric_list'][midx])
        tb_writer.add_scalar('Epoch_'+metric_name_list[midx]+'/val', epoch_metric, epoch_index)
        #change List item into float item
        info['epoch_metric_list'][midx] = epoch_metric
    #epoch time
    epoch_time = time.time() - epoch_start_time
    #make return info
    info['epoch_metric_dict'] = dict(zip(metric_name_list,info['epoch_metric_list']))
    info['epoch_loss'] = epoch_loss
    info['epoch_time'] = epoch_time
    return info






