#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午6:46
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_utils.py
# @IDE: PyCharm Community Edition
"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
from typing import List

import numpy as np
import tensorflow as tf
import os
import os.path as ops
import sys

from global_configuration import config
from local_utils import establish_char_dict,log_utils

logger = log_utils.init_logger()

class FeatureIO(object):
    """
        Implement the base writer class
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        # 没啥用，注视了
        # self.__char_dict = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        # self.__ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    @property
    def char_dict(self):
        """

        :return:
        """
        return self.__char_dict

    @staticmethod
    def int64_feature(value):
        """
            Wrapper for inserting int64 features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_int = True
        for val in value:
            if not isinstance(val, int):
                is_int = False
                value_tmp.append(int(float(val)))
        if is_int is False:
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        """
            Wrapper for inserting float features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_float = True
        for val in value:
            if not isinstance(val, int):
                is_float = False
                value_tmp.append(float(val))
        if is_float is False:
            value = value_tmp
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        """
            Wrapper for inserting bytes features into Example proto.
        """
        if not isinstance(value, bytes):
            if not isinstance(value, list):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def char_to_int(self, char: str) -> int:
        """

        :param char:
        :return:
        """
        temp = ord(char)
        # convert upper character into lower character
        if 65 <= temp <= 90:
            temp = temp + 32

        for k, v in self.__ord_map.items():
            if v == str(temp):
                temp = int(k)
                return temp
        raise KeyError("Character {} missing in ord_map.json".format(char))

        # TODO
        # Here implement a double way dict or two dict to quickly map ord and it's corresponding index

    def int_to_char(self, number: int) -> str:
        """ Return the character corresponding to the given integer.

        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """
        if number == '1':
            return '*'
        if number == 1:
            return '*'
        else:
            return self.__char_dict[str(number)]

    def encode_labels(self, labels):
        """
            encode the labels for ctc loss
        :param labels:
        :return:
        """
        encoded_labels = []
        lengths = []
        for label in labels:
            encode_label = [self.char_to_int(char) for char in label]
            encoded_labels.append(encode_label)
            lengths.append(len(label))
        return encoded_labels, lengths

    # 把返回的稀硫tensor，转化成对应的字符List
    '''
        标签序列,是一个稀疏矩阵SparseTensor,由3项组成：http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
        * indices: 二维int32的矩阵，代表非0的坐标点
        * values: 二维tensor，代表indice位置的数据值
        * dense_shape: 一维，代表稀疏矩阵的大小
        比如有3幅图，分别是123,4567,123456789那么
        indecs = [[0, 0], [0, 1], [0, 2], 
              [1, 0], [1, 1], [1, 2], [1, 3],
              [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]]
        values = [1, 2, 3 
              4, 5, 6, 7, 
              1, 2, 3, 4, 5, 6, 7, 8, 9]
        dense_shape = [3, 9]
        代表dense
        tensor:
        [[1, 2, 3, 0, 0, 0, 0, 0, 0]
        [4, 5, 6, 7, 0, 0, 0, 0, 0]
        [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    '''
    def sparse_tensor_to_str(self, sparse_tensor: tf.SparseTensor,characters) -> List[str]:
        """
        :param sparse_tensor: prediction or ground truth label
        :return: String value of the sparse tensor
        """
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        values = np.array([characters[id] for id in values])
        dense_shape = sparse_tensor.dense_shape

        # 先初始化一个2维矩阵，用['\n']来填充，因为这个字符不会出现在结果里面，可以当做特殊字符来处理
        # number_lists，实际上是一个dense向量
        number_lists = np.array([['\n'] * dense_shape[1]] * dense_shape[0], dtype=values.dtype)
        res = []

        #先把values，也就是有的值，拷贝到dense向量number_lists中
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]

        # 遍历这个dense的  number_list的每一行，变成一个字符数组
        for one_row in number_lists:
            res.append(''.join(c for c in one_row if c != '\n'))

        return res


class TextFeatureWriter(FeatureIO):
    """
        Implement the crnn feature writer
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        super(TextFeatureWriter, self).__init__(char_dict_path, ord_map_dict_path)
        return

    def write_features(self, tfrecords_path, labels, images, imagenames):
        """

        :param tfrecords_path:
        :param labels:
        :param images:
        :param imagenames:
        :return:
        """
        assert len(labels) == len(images) == len(imagenames)

        labels, length = self.encode_labels(labels)

        if not ops.exists(ops.split(tfrecords_path)[0]):
            os.makedirs(ops.split(tfrecords_path)[0])

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels': self.int64_feature(labels[index]),#"xxx"
                    'images': self.bytes_feature(image),        #byte[]
                    'imagenames': self.bytes_feature(imagenames[index])
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class TextFeatureReader(FeatureIO):
    """
        Implement the crnn feature reader
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        super(TextFeatureReader, self).__init__(char_dict_path, ord_map_dict_path)
        return

    @staticmethod
    def read_features(tfrecords_path, num_epochs):
        """

        :param tfrecords_path:
        :param num_epochs:
        :return:
        """
        assert ops.exists(tfrecords_path)

        filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'images': tf.FixedLenFeature((), tf.string),
                                               'imagenames': tf.FixedLenFeature([1], tf.string),
                                               'labels': tf.VarLenFeature(tf.int64),
                                           })
        image = tf.decode_raw(features['images'], tf.uint8)
        w, h = config.cfg.ARCH.INPUT_SIZE
        images = tf.reshape(image, [h, w, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames


class TextFeatureIO(object):
    """
        Implement a crnn feature io manager
    """
    def __init__(self, char_dict_path=ops.join(os.getcwd(), 'data/char_dict/char_dict.json'),#500个汉字
                 ord_map_dict_path=ops.join(os.getcwd(), 'data/char_dict/ord_map.json')):#？？？不知道干嘛用的
        """

        """
        self.__writer = TextFeatureWriter(char_dict_path, ord_map_dict_path)
        self.__reader = TextFeatureReader(char_dict_path, ord_map_dict_path)
        return

    @property
    def writer(self):
        """

        :return:
        """
        return self.__writer

    @property
    def reader(self):
        """

        :return:
        """
        return self.__reader
