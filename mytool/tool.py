# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月22日
"""

import tensorflow as tf
import functools
import tensorflow_text as tf_text
import os
from typing import *
from datetime import datetime
import pandas as pd
import yaml
import stat
import shutil
from tabulate import tabulate


###################################################################
###################################################################
#                         class
###################################################################
###################################################################


def import_class(module_path,class_name):
    """
    Dynamically imports one Class into namespace
    :param module_path: class module path
    :param class_name: class name must be the same as the origin
    :return:
        One Class definition
    """
    module = __import__(module_path,fromlist=[' '])
    return getattr(module,class_name)


###################################################################
###################################################################
#                         yaml
###################################################################
###################################################################


def load_yaml_args(yaml_filepath):
    """
    get args Dict
    :param yaml_filepath: str filepath
    :return:
        args: Dict
    """
    with open(yaml_filepath,encoding='utf-8') as f:
        args = yaml.load(f, yaml.Loader)
    return args


def save_yaml_args(yaml_filepath,data):
    with open(yaml_filepath,'w') as f:
        yaml.dump(data,f)


###################################################################
###################################################################
#                         func
###################################################################
###################################################################


def set_func_name(func,name):
    setattr(func,'__qualname__',name)
    return func


def get_func_name(func):
    """
    Gets the name of the function definition
    :param func:
    :return:
        func name
    """
    if hasattr(func,'__qualname__'):
        name_list = func.__qualname__.split('.')
        name = name_list[-1]
    else:
        name = func.__class__.__name__
    return name


###################################################################
###################################################################
#                         dict
###################################################################
###################################################################

def print_dicts_tablefmt(dicts:List[Dict], headers:Union[str,List[str]]='keys', tablefmt:str='fancy_grid'):
    """
    print the table of dicts.
    :param dicts: List[Dict] all dict must have same keys
    :param headers: first row of table
    :param tablefmt: args of tabulate. e.g. grid/pretty/presto/rst/orgtbl/github/simple
    :return:
        None
    """
    df = pd.DataFrame(dicts).T
    print(tabulate(df,headers=headers,tablefmt=tablefmt))


def modify_dict_with_hp(config_dict, hp, training=True):
    """
    modify config_dict with hyperparameters for searching of keras_tuner.
    :param config_dict: config dictionary
    :param hp: hyperparameters
    :return:
        modified config_dict
    """
    if not training: #best hp
        best_hp_dict = hp if isinstance(hp,dict) else hp.values

    for key in config_dict.keys():
        value = config_dict[key]
        if isinstance(value,dict):
            if 'type' in value.keys(): #需要修改
                if value['type'] == 'int':
                    step = value['step']  if 'step' in value.keys() else None
                    config_dict[key] = hp.Int(key,value['min'],value['max'],step) if training else best_hp_dict[key]
                elif value['type'] == 'float':
                    step = value['step'] if 'step' in value.keys() else None
                    config_dict[key] = hp.Float(key,value['min'],value['max'],step) if training else best_hp_dict[key]
                elif value['type'] == 'list':
                    config_dict[key] = hp.Choice(key,value['list']) if training else best_hp_dict[key]
                elif value['type'] == 'bool':
                    config_dict[key] = hp.Boolean(key) if training else best_hp_dict[key]
                else:
                    raise Exception('unknown type!!!')
            else:
                config_dict[key] = modify_dict_with_hp(value,hp,training)
        else: #一个数
            pass
    return config_dict


def config_dict_list_sort_with_argidx(config_dict_list, argidx):
    return_list = []
    for arg in argidx:
        return_list.append(config_dict_list[int(arg)])

    return return_list


def add_pre_or_suf_on_key_for_dict(obj_dict, prefix=None, suffix=None, checkpoint_str=None):
    """
    add prefix or suffix for every key of dict.
    :param obj_dict: dict
    :param prefix: str
    :param suffix: str
    :param checkpoint_str: str if key contains checkpoint_str, it will be modified.
    :return:
        return_dict
    """
    if prefix is None and suffix is None:
        raise Exception('prefix or suffix must have str value')

    return_dict = {}

    if prefix is not None:
        for key in obj_dict.keys():
            if checkpoint_str is not None:
                if key.find(checkpoint_str) != -1:
                    if suffix is not None: #both add
                        return_dict[prefix + key + suffix] = obj_dict[key]
                    else: #add prefix
                        return_dict[prefix + key] = obj_dict[key]
                else:
                    return_dict[key] = obj_dict[key]
            else:
                if suffix is not None: #both add
                    return_dict[prefix + key + suffix] = obj_dict[key]
                else: #add prefix
                    return_dict[prefix + key] = obj_dict[key]
    else: #suffix is not None
        for key in obj_dict.keys():
            if checkpoint_str is not None:
                if key.find(checkpoint_str) != -1: #add suffix
                    return_dict[key + suffix] = obj_dict[key]
                else:
                    return_dict[key] = obj_dict[key]
            else:#add suffix
                return_dict[key + suffix] = obj_dict[key]

    return return_dict


def complete_dict_path(dict_args:Dict,work_dir:str,check_suffix:str='path'):
    """
    complete relative path in a config dictionary with work_dir
    :param dict_args: config dictionary
    :param work_dir: path which will be added to prefix of relative path
    :param check_suffix: key which ends with it to complete
    :return:
        completed config dictionary
    """
    return_dict = {}
    for key,value in dict_args.items():
        if key.endswith(check_suffix):
            return_dict[key] = os.path.join(work_dir,value)
        else:
            return_dict[key] = value
    return return_dict


def remove_dict_None_value(dict_args):
    """
    remove None value from Dict
    :param dict_args: Dict
    :return:
        dict_args: Dict
    """
    keys = dict_args.keys()
    remove_keys = []
    for key in keys:
        if dict_args[key] == None:
            remove_keys.append(key)
    for key in remove_keys:
        dict_args.pop(key)
    return dict_args


###################################################################
###################################################################
#                         os
###################################################################
###################################################################


def remove_prefile_in_dir(dir, num_check_threshold, remove_ratio):
    allfilenames = os.listdir(dir)
    allfilepath = [os.path.join(dir, filename) for filename in allfilenames]
    if len(allfilepath) > num_check_threshold:
        allfilepath.sort()  # 从小到大 从旧到新
        # 删除最旧的几个
        need_remove_num = int(remove_ratio * len(allfilepath))
        for i in range(need_remove_num):
            os.chmod(allfilepath[i], stat.S_IWRITE)
            if os.path.isfile(allfilepath[i]):
                os.remove(allfilepath[i])
            elif os.path.isdir(allfilepath[i]):
                shutil.rmtree(allfilepath[i])
            else:
                raise Exception('Unknown Error!')
    else:
        pass


def get_basename_split_ext(path):
    """
    get basename except ext.
    :param path: str
    :return:
        basename
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]


###################################################################
###################################################################
#                         datetime
###################################################################
###################################################################


def get_datetime_name():
    """
    get datetime name for save configuration
    :return:
        timestamp: datetime name
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    return timestamp


def get_date_range_start_end_pd(start_time_str,end_time_str,freq):
    """
    :param start_time_str: time_str
    :param end_time_str: time_str
    :param freq: pd.Timedelta e.g. 1D 1W 1min
        Y: 年(末)
        Q: 季度(末)
        M: 月(末)
        W: 星期
        D: 天
        B: 工作日
        H: 小时
        T(min): 分钟
        S: 秒
        L(ms): 毫秒
        U(us): 微秒
        N: 纳秒
    :return:
        date_list: 1-D [start_time_str:end_time_str:freq]
    other_applications:
        date_list.dayofyear .weekofyear 一年中第几天(从1算) 第几个星期(从1算，但可能有去年星期)
                .dayofweek .weekday 一星期中第几天(0星期一，6星期日)
        more information url:https://pandas.pydata.org/docs/reference/arrays.html#datetime-data
    """
    return pd.date_range(start_time_str,end_time_str,freq=freq)


def get_date_range_start_step_pd(start_time_str,freq,steps):
    """
    :param start_time_str: time_str
    :param freq: pd.Timedelta e.g. 1D 1W 1min
        Y: 年(末)
        Q: 季度(末)
        M: 月(末)
        W: 星期
        D: 天
        B: 工作日
        H: 小时
        T(min): 分钟
        S: 秒
        L(ms): 毫秒
        U(us): 微秒
        N: 纳秒
    :param steps: int
    :return:
        date_list: 1-D [start_time_str::freq] len=steps
    """
    return pd.date_range(start_time_str,periods=steps,freq=freq)


def get_latest_datetime_checkpoint(checkpoint_dir):
    """
    get latest checkpoint name from given checkpoint_dir
    :param checkpoint_dir: file to save
    :return:
        latest file name
    """
    filename_list = os.listdir(checkpoint_dir)
    filename_list.sort() #小 -> 大
    return os.path.join(checkpoint_dir,filename_list[-1])


###################################################################
###################################################################
#                         Text
###################################################################
###################################################################

def map_bytes_to_str(x_or_li):
    """
    将bytes或List[bytes]解码
    :param x_or_li: bytes or List[bytes]
    :return:
        x_or_li: decoded
    """
    if isinstance(x_or_li,bytes):
        return x_or_li.decode()
    elif isinstance(x_or_li,list):
        return list(map(map_bytes_to_str,x_or_li))
    else:
        raise Exception('must contain bytes!!')


###################################################################
###################################################################
#                         Tensorflow
###################################################################
###################################################################

#自定义类的初始化函数的参数都当做是python原始类型
#如果要用这些参数参与tf.Tensor计算,要有显示的转换为Tensor
#因为可能get_config都是返回python原始类型给初始化函数
def convert_to_py(x_or_list_or_tensor,list_contain_tensor=False,if_print_track=False):
    """
    将tensorflow的tensor转换为python类型
    :param x_or_list_or_tensor: tensor or List[tensor] or python_local_types
    :param list_contain_tensor: if List[tensor] must be True
    :param if_print_track: whether to print the track of x_or_list_or_tensor
    :return:
        x_or_list_or_tensor: types converted
    """
    if if_print_track: print('1',x_or_list_or_tensor)
    if isinstance(x_or_list_or_tensor,(int,float,bool,complex,str,list,tuple,dict,set)):
        if list_contain_tensor:
            if len(x_or_list_or_tensor) == 0: #空 []
                if if_print_track: print('2',[])
                return []
            else: #len > 0
                if tf.is_tensor(x_or_list_or_tensor): #[tensor]
                    if if_print_track: print('3',x_or_list_or_tensor)
                    return list(map(functools.partial(convert_to_py,list_contain_tensor=False,if_print_track=if_print_track),x_or_list_or_tensor))
                else:
                    if if_print_track: print('4',x_or_list_or_tensor) #[[tensor]]至少二层list
                    return list(map(functools.partial(convert_to_py,list_contain_tensor=True,if_print_track=if_print_track),x_or_list_or_tensor))
        else:
            if if_print_track: print('5',x_or_list_or_tensor)
            return x_or_list_or_tensor

    elif tf.is_tensor(x_or_list_or_tensor):
        if x_or_list_or_tensor.dtype == tf.string: #bytes
            x_or_list_or_tensor = x_or_list_or_tensor.numpy()
            if isinstance(x_or_list_or_tensor,bytes): #one bytes
                if if_print_track: print('6',x_or_list_or_tensor)
                return x_or_list_or_tensor.decode()
            else: #bytes list
                x_or_list_or_tensor = x_or_list_or_tensor.tolist()
                if if_print_track: print('7',x_or_list_or_tensor)
                return map_bytes_to_str(x_or_list_or_tensor)
        else: #number
            if if_print_track: print('8',x_or_list_or_tensor)
            return x_or_list_or_tensor.numpy().tolist()
    else:
        raise Exception('TypeError!!')


#使用tf_text.pad_model_inputs，输入只能是ragged，只裁剪第二维
#返回值后两个，第二个是mask
def pad_or_cut_matrix(mt,max_size,axis=0,pad_value=0):
    """
    pad or cut tensor in axis to max_size with pad_value
    :param mt: tensor
    :param max_size: integer
    :param axis: integer
    :param pad_value: numeric
    :return:
        f_reperm: padd or cut tensor
    """
    mt_shape = tf.shape(mt)
    target_size = mt_shape[axis]
    remain_shape = tf.concat([mt_shape[:axis], mt_shape[axis+1:]],axis=0)
    ori_perm = tf.range(tf.rank(mt),dtype=tf.int32)
    perm = tf.concat([ori_perm[:axis], ori_perm[axis+1:], tf.convert_to_tensor([ori_perm[axis]],dtype=tf.int32)],axis=0)
    reperm = tf.math.invert_permutation(perm)
    b_transpose = tf.transpose(mt,perm)
    c_rank2_mt = tf.reshape(b_transpose,[-1,target_size])
    d_rank2_pad_or_cut,_ = tf_text.pad_model_inputs(tf.RaggedTensor.from_tensor(c_rank2_mt),max_size,pad_value)
    e_reshape = tf.reshape(d_rank2_pad_or_cut,tf.concat([remain_shape, tf.convert_to_tensor([max_size],dtype=tf.int32)],axis=0))
    f_reperm = tf.transpose(e_reshape,reperm)
    return f_reperm


def get_random_idx(need_idx_len,contain_max_idx_val_from_zero=None):
    """
    get random index from 0 to contain_max_idx_val_from_zero(contain)
    :param need_idx_len: integer 需要多长的shuffle后的idx
    :param contain_max_idx_val_from_zero: integer 允许出现的最大下标是什么(包含这个最大值)
    :return:
        idx_list: 1-D
    """
    if contain_max_idx_val_from_zero is None:
        contain_max_idx_val_from_zero = need_idx_len - 1
    seed = tf.random.uniform([3,],maxval=tf.cast(tf.timestamp(),tf.int32),dtype=tf.int32)
    return tf.random_index_shuffle(
        index=tf.range(need_idx_len,dtype=tf.int32),
        seed=seed,
        max_index=contain_max_idx_val_from_zero)


def random_matrix(mt,axis=0):
    """
    shuffle tensor in axis
    :param mt: tensor
    :param axis: integer
    :return:
        shuffle tensor
    """
    axis_len = tf.shape(mt)[axis]
    random_idx = get_random_idx(axis_len)
    return tf.gather(mt,random_idx,axis=axis)


if __name__ == '__main__':
    pass