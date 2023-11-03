# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月22日
"""

import tensorflow as tf
import numpy as np


def cal_attention(Q, K, V, mask=None, return_score=False):
    """
    计算Q、K、V注意力
    :param Q: [bq,nq,dq]
    :param K: [bk,nk,dk]
    :param V: [bv,nv,dv]
    :param mask: [nq,nk] float
    :param return_score: [] bool 是否返回计算的注意力权重
    :return:
        output: [b,nq,dv] 注意力输出
        A: [b,nq,nk] 注意力权重
    :limit
        bq == bk == bv == b
        dq == dk
        nk == nv
    """
    bq, nq, dq = tf.shape(Q)
    bk, nk, dk = tf.shape(K)
    bv, nv, dv = tf.shape(V)
    assert bq == bk == bv
    assert dq == dk
    assert nk == nv
    # Matmul
    U = tf.einsum("bni,bid->bnd", Q, tf.transpose(K, [0, 2, 1]))
    # Scale
    U /= tf.sqrt(tf.cast(dk, dtype=tf.float32))  # or dk==dq
    # Mask
    if mask is not None:
        bm, nm, dm = tf.shape(mask)
        assert bm == bq
        assert nm == nq
        assert dm == nk
        U *= mask
    # softmax(score)
    A = tf.nn.softmax(U, axis=-1)
    ba, na, da = tf.shape(A)
    assert ba == bq
    assert na == nq
    assert da == nk
    # output
    output = tf.einsum("bni,bid->bnd", A, V)
    if return_score:
        return output, A
    return output


class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, key_dim, value_dim=None, **kwargs):
        super().__init__(**kwargs)
        """
        key_dim == dq == dk
        value_dim == dv
        """
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

    def build(self, input_shape):
        """
        first_value(shape): Q
        second_value(shape): V
        third_value(shape): K
        fourth_value: mask
        """
        self.Q_dense = tf.keras.layers.Dense(self.n_heads * self.key_dim)
        self.K_dense = tf.keras.layers.Dense(self.n_heads * self.key_dim)
        self.V_dense = tf.keras.layers.Dense(self.n_heads * self.key_dim)
        if self.value_dim is not None and len(input_shape) > 2:
            self.V_dense = tf.keras.layers.Dense(self.n_heads * self.value_dim)
        self.final_dense = tf.keras.layers.Dense(input_shape[0][-1])

    def call(self, inputs, return_score=False):
        """
        args:
            inputs:
                bq == bk == bv
                dq == dk
                nk == nv
                mask must be [batch, nq, nk]
                first_value: Q
                second_value: V
                third_value: K
                fourth_value: mask
            return_score: 是否返回scores [bq, n_heads, nq, nk]

        return:
            output: 返回的output.shape一定等同于Q
            scores: [bq, n_heads, nq, nk]

        example:
            Q = tf.random.normal((5,3,8))
            K = tf.random.normal((5,4,8))
            V = tf.random.normal((5,4,6))
            mask = np.random.randint(2,size=(5,3,4))
            cu = CustomMultiHeadAttention(2,12)
            cu([Q,V,K,mask],return_score=True)
        """
        Q = inputs[0]
        mask = None
        if len(inputs) == 1:
            K = V = Q
        elif len(inputs) == 2:
            K = V = inputs[1]
        elif len(inputs) == 3:
            V = inputs[1]
            K = inputs[2]
        elif len(inputs) == 4:
            V = inputs[1]
            K = inputs[2]
            mask = inputs[3]
        else:
            raise Exception("len(inputs) in call() must be less than 5 more than 0")

        bq, nq, dq = tf.shape(Q)
        bk, nk, dk = tf.shape(K)
        bv, nv, dv = tf.shape(V)

        Qd = self.Q_dense(Q)
        Kd = self.K_dense(K)
        Vd = self.V_dense(V)
        # reshape and transpose
        # (batch, n, n_heads * d) -> (batch, n, n_heads, d)
        # -> (n_heads, batch, n, d) -> (n_heads*batch, n, d)
        Qd = tf.reshape(Qd, [bq, nq, self.n_heads, self.key_dim])  # 2
        Qd = tf.transpose(Qd, [2, 0, 1, 3])  # 3
        Qd = tf.reshape(Qd, [self.n_heads * bq, nq, self.key_dim])  # 4

        Kd = tf.reshape(Kd, [bk, nk, self.n_heads, self.key_dim])  # 2
        Kd = tf.transpose(Kd, [2, 0, 1, 3])  # 3
        Kd = tf.reshape(Kd, [self.n_heads * bk, nk, self.key_dim])  # 4

        Vd = tf.reshape(Vd, [bv, nv, self.n_heads, self.value_dim or self.key_dim])  # 2
        Vd = tf.transpose(Vd, [2, 0, 1, 3])  # 3
        Vd = tf.reshape(Vd, [self.n_heads * bv, nv, self.value_dim or self.key_dim])  # 4

        if mask is not None:
            mask = tf.cast(tf.repeat(mask, self.n_heads, axis=0), dtype=tf.float32)

        # cal_attention
        if return_score:
            output, scores = cal_attention(Qd, Kd, Vd, mask, return_score)
        else:
            output = cal_attention(Qd, Kd, Vd, mask, return_score)
        # to (batch, nq, n_heads * dv)
        output = tf.reshape(output, [self.n_heads, bq, nq, self.value_dim or self.key_dim])  # 3
        output = tf.transpose(output, [1, 2, 0, 3])  # 2
        output = tf.reshape(output, [bq, nq, self.n_heads * (self.value_dim or self.key_dim)])  # 1
        # to (batch, nq, dq)
        output_d = self.final_dense(output)
        assert tf.shape(output_d)[-1] == dq

        if return_score:
            return output, tf.reshape(scores, [bq, self.n_heads, nq, nk])

        return output_d





#STN
def affine_grid(batch_theta, batch_output_shape):
    """
    batch_theta:
        shape: [b,2,3]
    batch_output_shape:
        value: [b,oh,ow,c]

    return:
        batch_affine_grid:
            shape: [b,oh,ow,2]
    """
    # common data
    oh = batch_output_shape[1]
    ow = batch_output_shape[2]
    oh_max = 1 - 1 / oh  # value_range: [0,1]
    oh_min = -oh_max  # value_range: [-1,0]
    ow_max = 1 - 1 / ow
    ow_min = -ow_max
    # [oh,]
    oh_lim = tf.cast(tf.linspace(oh_min, oh_max, oh), dtype=tf.float32)
    # [ow,]
    ow_lim = tf.cast(tf.linspace(ow_min, ow_max, ow), dtype=tf.float32)
    # [oh,ow] [oh,ow]
    h_mt, w_mt = tf.meshgrid(oh_lim, ow_lim, indexing='ij')
    # [oh,ow,3]
    position_hw1 = tf.concat(
        [h_mt[..., tf.newaxis], w_mt[..., tf.newaxis], tf.ones_like(h_mt, dtype=tf.float32)[..., tf.newaxis]], axis=-1)
    # [b,oh,ow,3]
    batch_position_hw1 = tf.tile(position_hw1[tf.newaxis, ...], [batch_output_shape[0], 1, 1, 1])
    # [b,3,2]
    batch_theta_transpose = tf.transpose(batch_theta, [0, 2, 1])
    # [b,oh,ow,2]                                     [b,oh,ow,3]        [b,3,2]
    batch_affine_grid = tf.einsum('bhwx,bxn -> bhwn', batch_position_hw1, batch_theta_transpose)

    return batch_affine_grid


def grid_sample(batch_input, batch_affine_grid):
    """
    method: bilinear
    batch_input:
        shape: [b,ih,iw,c]
    batch_affine_grid:
        shape: [b,oh,ow,2]

    return:
        batch_result_image:
            shape: [b,oh,ow,c]
    """
    # 获取原始图像(batch_input)位置网格
    # [4,] value:[b,ih,iw,c]
    batch_input_shape = tf.shape(batch_input)
    # common data
    ih, iw = batch_input_shape[1], batch_input_shape[2]
    # [ih,iw] [ih,iw]
    h_mt, w_mt = tf.meshgrid(tf.range(batch_input_shape[1], dtype=tf.float32),
                             tf.range(batch_input_shape[2], dtype=tf.float32), indexing='ij')
    # [ih,iw,2]
    position_hw = tf.concat([h_mt[..., tf.newaxis], w_mt[..., tf.newaxis]], axis=-1)

    # 先归一化batch_affine_grid再规范到原始图像(batch_input)大小
    # [4,] [b,oh,ow,2]
    batch_affine_grid_shape = tf.shape(batch_affine_grid)
    # common data
    oh, ow = batch_affine_grid_shape[1], batch_affine_grid_shape[2]
    oh_max, ow_max = tf.cast(1 - 1 / oh, dtype=tf.float32), tf.cast(1 - 1 / ow, dtype=tf.float32)
    oh_min, ow_min = -oh_max, -ow_max
    # [b,oh,ow,2] 归一化                     [2,]                                                    [2,]
    batch_affine_grid = (batch_affine_grid - tf.convert_to_tensor([oh_min, ow_min],
                                                                  dtype=tf.float32)) / tf.convert_to_tensor(
        [oh_max - oh_min, ow_max - ow_min], dtype=tf.float32)
    # [b,oh,ow,2] 规范到原始图像(batch_input)大小 [2,]
    batch_affine_grid = batch_affine_grid * tf.convert_to_tensor([ih - 1, iw - 1], dtype=tf.float32)

    # 计算各网格点像素值
    # method1:计算量小
    # batch_affine_grid中的值代表在原图中的位置[h,w]
    # [[w1h2,w2h2],
    # [w1h1,w2h1]]
    h = batch_affine_grid[..., 0:1]  # [b,oh,ow,1]
    w = batch_affine_grid[..., 1:2]  # [b,oh,ow,1]

    h1 = tf.cast(tf.floor(batch_affine_grid[..., 0:1]), dtype=tf.int32)  # [b,oh,ow,1]
    h2 = h1 + 1  # [b,oh,ow,1]
    w1 = tf.cast(tf.floor(batch_affine_grid[..., 1:2]), dtype=tf.int32)  # [b,oh,ow,1]
    w2 = w1 + 1  # [b,oh,ow,1]

    h1 = tf.clip_by_value(h1, 0, ih - 1)  # [b,oh,ow,1]
    h2 = tf.clip_by_value(h2, 0, ih - 1)  # [b,oh,ow,1]
    w1 = tf.clip_by_value(w1, 0, iw - 1)  # [b,oh,ow,1]
    w2 = tf.clip_by_value(w2, 0, iw - 1)  # [b,oh,ow,1]

    # get pixel value
    h1w1 = tf.concat([h1, w1], axis=-1)  # [b,oh,ow,2]
    h1w2 = tf.concat([h1, w2], axis=-1)  # [b,oh,ow,2]
    h2w1 = tf.concat([h2, w1], axis=-1)  # [b,oh,ow,2]
    h2w2 = tf.concat([h2, w2], axis=-1)  # [b,oh,ow,2]
    #                            [b,ih,iw,c] [b,oh,ow,2]
    fh1w1 = tf.cast(tf.gather_nd(batch_input, h1w1, batch_dims=1), dtype=tf.float32)  # [b,oh,ow,c]
    fh1w2 = tf.cast(tf.gather_nd(batch_input, h1w2, batch_dims=1), dtype=tf.float32)  # [b,oh,ow,c]
    fh2w1 = tf.cast(tf.gather_nd(batch_input, h2w1, batch_dims=1), dtype=tf.float32)  # [b,oh,ow,c]
    fh2w2 = tf.cast(tf.gather_nd(batch_input, h2w2, batch_dims=1), dtype=tf.float32)  # [b,oh,ow,c]

    # method1-1
    h1 = tf.cast(h1, dtype=tf.float32)  # [b,oh,ow,1]
    h2 = tf.cast(h2, dtype=tf.float32)  # [b,oh,ow,1]
    w1 = tf.cast(w1, dtype=tf.float32)  # [b,oh,ow,1]
    w2 = tf.cast(w2, dtype=tf.float32)  # [b,oh,ow,1]
    # [b,oh,ow,c]
    fP = (h2 - h) * (w2 - w) * fh1w1 + (h2 - h) * (w - w1) * fh1w2 + (h - h1) * (w2 - w) * fh2w1 + (h - h1) * (
                w - w1) * fh2w2

    # #method1-2:占用存储空间大(比method1-1大了4*2-4*c倍)
    # h1w1 = tf.cast(h1w1,dtype=tf.float32)
    # h1w2 = tf.cast(h1w2,dtype=tf.float32)
    # h2w1 = tf.cast(h2w1,dtype=tf.float32)
    # h2w2 = tf.cast(h2w2,dtype=tf.float32)
    # #[b,oh,ow,4,2]
    # grid4 = tf.stack([h1w1,h1w2,h2w1,h2w2],axis=3)
    # #[b,oh,ow,4,c]
    # fgrid4 = tf.stack([fh1w1,fh1w2,fh2w1,fh2w2],axis=3)
    # #[b,oh,ow,4,2]                           [b,oh,ow,1,2]                         [b,oh,ow,4,2]
    # distance_bhw42 = tf.maximum(0,1 - tf.abs(batch_affine_grid[...,tf.newaxis,:] - grid4))
    # #[b,oh,ow,4,c]     [b,oh,ow,4,1]             [b,oh,ow,4,1]             [b,oh,ow,4,c]
    # batch_dist_input = distance_bhw42[...,0:1] * distance_bhw42[...,1:2] * fgrid4
    # #[b,oh,ow,c]
    # fP = tf.reduce_sum(batch_dist_input,axis=[3])

    return fP

    # method2:占用存储空间大(比method1-1大了ih*iw*2-ih*iw*c倍)
    # #计算位置距离再计算像素值
    # #[b,oh,ow,ih,iw,2]                        [b,oh,ow,1,1,2]                                  [1,1,1,ih,iw,2]
    # distance_bhwhw2 = tf.maximum(0,1 - tf.abs(batch_affine_grid[...,tf.newaxis,tf.newaxis,:] - position_hw[tf.newaxis,tf.newaxis,tf.newaxis,...]))
    # #[b,oh,ow,ih,iw,c] [b,oh,ow,ih,iw,1]          [b,oh,ow,ih,iw,1]          [b,1,1,ih,iw,c]
    # batch_dist_input = distance_bhwhw2[...,0:1] * distance_bhwhw2[...,1:2] * batch_input[:,tf.newaxis,tf.newaxis,...]
    # #[b,oh,ow,c]
    # batch_result_image = tf.reduce_sum(batch_dist_input,axis=[3,4])
    # return batch_result_image


def positional_encoding(length, depth):
    """
    如果嵌入后的shape为(batch, n, d)
    lenght: n
    depth: d

    return: tensor (length, depth)
    """
    depth_copy = depth
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (seq, depth)

    sin_angle_rads = np.sin(angle_rads)  # (seq, depth)
    cos_angle_rads = np.cos(angle_rads)  # (seq, depth)

    pos_encoding = np.stack([sin_angle_rads, cos_angle_rads], axis=2)  # [seq,depth,2]
    pos_encoding = np.reshape(pos_encoding, (length, depth_copy))  # [seq,2*depth] or [seq,depth_copy]

    return tf.cast(pos_encoding, dtype=tf.float32)










