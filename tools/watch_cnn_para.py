#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-13 下午1:22
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : watch_cnn_para.py.py
# @IDE: PyCharm Community Edition
"""
Watch the cnn parameters
"""
import os
import os.path as ops
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from crnn_model import crnn_model
from global_configuration import config


inputdata = tf.placeholder(dtype=tf.float32, shape=[32, 32, 100, 3], name='input')
shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25,
                                 num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='gru')

with tf.variable_scope('shadow', reuse=False):
    net_out = shadownet.build_shadownet(inputdata=inputdata)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess=sess, save_path='/home/baidu/CRNN_Tensorflow/model/shadownet/two_character/'
                                       'shadownet_two_character_2017-12-13-09-31-14.ckpt-6000')

    var_list = np.array(sess.run(tf.trainable_variables()))

    conv1_1 = var_list[0][:, :, :, 0]
    conv1_1 = np.asarray((conv1_1 - np.min(conv1_1))/(np.max(conv1_1) - np.min(conv1_1))*255).astype(np.uint8)

    plt.imshow(conv1_1)
    plt.show()

    print('Complete')
