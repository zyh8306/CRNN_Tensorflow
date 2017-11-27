#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-23 下午4:20
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet_subnet.py.py
# @IDE: PyCharm Community Edition
"""
Train shadow net CNN sub network
"""
import argparse
import os
import os.path as ops
import sys
import time

import numpy as np
import tensorflow as tf
import pprint

sys.path.append(os.getcwd())

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config


logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--weights_path', type=str, help='Where you store the pretrained weights')

    return parser.parse_args()


def train_shadownet(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    images, labels, imagenames = decoder.read_features(dataset_dir, num_epochs=None, flag='Train')
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)

    # initializa the net model
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25,
                                     num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet_cnn_subnet(inputdata=inputdata)

    correct_preds = tf.equal(tf.argmax(tf.nn.softmax(net_out), 1),
                             tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy_train')

    print(net_out.get_shape().as_list())
    print(input_labels.get_shape().as_list())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=net_out, name='cost'))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Accuracy', tensor=accuracy)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_cnnsub_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = config.cfg.TRAIN.EPOCHS

    print('Global configuration is as follows:')
    pprint.pprint(config.cfg)

    with sess.as_default():

        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(train_epochs):
            _, c, train_out, train_accuracy, summary = sess.run(
                [optimizer, cost, net_out, accuracy, merge_summary_op])

            summary_writer.add_summary(summary=summary, global_step=epoch)

            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:03d} batch: {:d} cost= {:9f} training accuracy= {:9f}'.format(
                    epoch + 1, epoch + 1, c, train_accuracy))

            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if not ops.exists(args.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))

    train_shadownet(args.dataset_dir, args.weights_path)
    print('Done')
