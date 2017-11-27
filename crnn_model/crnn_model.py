#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-21 下午6:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : crnn_model.py
# @IDE: PyCharm Community Edition
"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
import tensorflow as tf
from tensorflow.contrib import rnn

from crnn_model import cnn_basenet
from global_configuration import config

cfg = config.cfg


class ShadowNet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    def __init__(self, phase, hidden_nums, layers_nums, seq_length, num_classes, rnn_cell_type='lstm'):
        """

        :param phase:
        :param hidden_nums:
        :param layers_nums:
        :param seq_length:
        :param num_classes:
        :param rnn_cell_type:
        """
        super(ShadowNet, self).__init__()
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__seq_length = seq_length
        self.__num_classes = num_classes
        self.__rnn_cell_type = rnn_cell_type.lower()
        if self.__rnn_cell_type not in ['lstm', 'gru']:
            raise ValueError('rnn_cell_type should be in [\'lstm\', \'gru\']')
        return

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
        return

    def __conv_stage(self, inputdata, out_dims, name=None):
        """
        Traditional conv stage in VGG format
        :param inputdata:
        :param out_dims:
        :return:
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
        if self.phase.lower() == 'train':
            conv_bn = self.layerbn(conv, is_training=True)
        else:
            conv_bn = self.layerbn(conv, is_training=False)
        relu = self.relu(inputdata=conv_bn)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def __feature_sequence_extraction(self, inputdata):
        """
        Implement the 2.1 Part Feature Sequence Extraction
        :param inputdata: eg. batch*32*100*3 NHWC format
        :return:
        """
        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64
        conv2 = self.__conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128
        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3') # batch*8*25*256
        if self.phase.lower() == 'train':
            conv3_bn = self.layerbn(conv3, is_training=True)
        else:
            conv3_bn = self.layerbn(conv3, is_training=False)
        relu3 = self.relu(conv3_bn) # batch*8*25*256
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*25*256
        if self.phase.lower() == 'train':
            conv4_bn = self.layerbn(conv4, is_training=True)
        else:
            conv4_bn = self.layerbn(conv4, is_training=False)
        relu4 = self.relu(conv4_bn)  # batch*8*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*25*256
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*25*512
        if self.phase.lower() == 'train':
            conv5_bn5 = self.layerbn(inputdata=conv5, is_training=True)
        else:
            conv5_bn5 = self.layerbn(inputdata=conv5, is_training=False)  # batch*4*25*512
        relu5 = self.relu(conv5_bn5)  # batch*4*25*512
        conv6 = self.conv2d(inputdata=relu5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6')  # batch*4*25*512
        if self.phase.lower() == 'train':
            conv6_bn6 = self.layerbn(inputdata=conv6, is_training=True)
        else:
            conv6_bn6 = self.layerbn(inputdata=conv6, is_training=False)  # batch*4*25*512
        relu6 = self.relu(conv6_bn6)  # batch*4*25*512
        max_pool6 = self.maxpooling(inputdata=relu6, kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512
        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7')  # batch*1*25*512
        relu7 = self.relu(conv7)  # batch*1*25*512
        return relu7

    def __map_to_sequence(self, inputdata):
        """
        Implement the map to sequence part of the network mainly used to convert the cnn feature map to sequence used in
        later stacked lstm layers
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata):
        """
        Implement the sequence label part of the network
        :param inputdata:
        :return:
        """
        if self.__rnn_cell_type == 'lstm':
            with tf.variable_scope('LSTMLayers'):
                # construct stack lstm rcnn layer
                # forward lstm cell
                fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums, self.__hidden_nums]]
                # Backward direction cells
                bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums, self.__hidden_nums]]

                stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                             dtype=tf.float32)

                if self.phase.lower() == 'train':
                    stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)

                [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
                rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

                w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
                # Doing the affine projection

                logits = tf.matmul(rnn_reshaped, w)

                logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

                raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

                # Swap batch and batch axis
                rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]
        else:
            with tf.variable_scope('GRULayers'):
                # construct stack lstm rcnn layer
                # forward lstm cell
                fw_cell_list = [rnn.GRUCell(nh) for nh in
                                [self.__hidden_nums, self.__hidden_nums]]
                # Backward direction cells
                bw_cell_list = [rnn.GRUCell(nh) for nh in
                                [self.__hidden_nums, self.__hidden_nums]]

                stack_gru_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                            dtype=tf.float32)

                if self.phase.lower() == 'train':
                    stack_gru_layer = self.dropout(inputdata=stack_gru_layer, keep_prob=0.5)

                [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
                rnn_reshaped = tf.reshape(stack_gru_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

                w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
                # Doing the affine projection

                logits = tf.matmul(rnn_reshaped, w)

                logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

                raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

                # Swap batch and batch axis
                rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def build_shadownet(self, inputdata):
        """

        :param inputdata:
        :return:
        """
        with tf.variable_scope('cnn_subnetwork'):
            # first apply the cnn feature extraction stage
            cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

            # second apply the map to sequence stage
            sequence = self.__map_to_sequence(inputdata=cnn_out)

            # third apply the sequence label stage
            net_out, raw_pred = self.__sequence_label(inputdata=sequence)

        return net_out

    def build_shadownet_cnn_subnet(self, inputdata):
        """

        :param inputdata:
        :return:
        """
        # first apply the cnn feture extraction stage
        with tf.variable_scope('cnn_subnetwork'):
            cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

            fc1 = self.fullyconnect(inputdata=cnn_out, out_dim=4096, use_bias=False, name='fc1')

            relu1 = self.relu(inputdata=fc1, name='relu1')

            fc2 = self.fullyconnect(inputdata=relu1, out_dim=cfg.TRAIN.CLASSES_NUMS, use_bias=False, name='fc2')

        return fc2
