#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午3:59
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : cnn_basenet.py
# @IDE: PyCharm Community Edition
"""
The base convolution neural networks mainly implement some useful cnn functions
"""
import tensorflow as tf
import numpy as np
from abc import ABCMeta 
#https://docs.python.org/zh-cn/3/library/abc.html  抽象基类 (ABC) 的组件
#Python本身不提供抽象类和接口机制，要想实现抽象类，可以借助abc模块。ABC是Abstract Base Class的缩写。
#http://yansu.org/2013/06/09/learn-python-abc-module.html


class CNNBaseModel(metaclass=ABCMeta):
    """
    Base model for other specific cnn ctpn_models
    """
    def __init__(self):
        pass

    # 这个方法定义一个卷基层
    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME', stride=1, w_init=None, b_init=None,
               nl=tf.identity, split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel. #输出的channel，其实就是对应多少个核
        :param kernel_size: int so only support square kernel convolution，如果是一个数，会自动变成 数x数，比如3=> 3x3
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param nl: a tensorflow identify function
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            # in_channel，看，输入的channel数，是根据输入数据自动得到的，定义卷积核的输入channel是基于输入数据的
            # 这个规则，下面会详细介绍
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                # [3,3,input Channel,output channel]
                # 这里有个细节，这个卷积核的输入channel必须是上一个tensor的channel数
                # 比如输入是 [200W,300H,256C]的数据，那么卷积核就得是[3W,3H,256C]，这样才可以和之前的数据匹配上，卷积一次产生一个点
                # 所以，一个卷积核产生一张feature map，这个时候，就需要定义卷积核的数量，也就是这里的out_channel，
                # 比如，我定义卷积核数量（out_channel)是512，那么[200W,300H,256C]的输入和512个[3W,3H,256C]卷积核卷积后，
                # 就得到了[200W,300H,512C]的结果啦。卷积核就是我们要学的参数。
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None  #filter_shape:3,3,3,64 => (kernel_size,kernel_size,in_channel, out_channel),//out_channel其实就是隐藏层的神经元个数啊

            if use_bias: # 注意这里还有bias，个数是一个卷积核对一个b，默认是有的
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                #定义一个conv2d的卷积层出来了，这个conv就是卷积核算出的结果了
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                #split是Alexnet用的，Alexnet是分叉的，我们默认不是
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            #加上偏置bias
            ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True, data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, noise_shape=None, name=None):
        """

        :param name:
        :param inputdata:
        :param keep_prob:
        :param noise_shape:
        :return:
        """
        return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None, nl=tf.identity, use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param nl: a nonlinearity function
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: nl(x, name='output'), use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init, trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training):
        """

        :param inputdata:
        :param is_training:
        :return:
        """
        output = tf.contrib.layers.batch_norm(inputdata, scale=True, is_training=is_training, updates_collections=None)
        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)
