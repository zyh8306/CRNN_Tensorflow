#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config


logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Path to dir containing train/test data and annotation files.')
    parser.add_argument('-w', '--weights_path', type=str, help='Path to pre-trained weights.')
    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
                        help='Number of threads to use in batch shuffling')

    return parser.parse_args()


def train_shadownet(dataset_dir, weights_path=None, num_threads=4):
    """

    :param dataset_dir:
    :param weights_path:
    :param num_threads: Number of threads to use in tf.train.shuffle_batch
    :return:
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    #使用的是tensorflow的tfrecords格式，预加载文件形式，直接load到内存，速度快
    images, labels, imagenames = decoder.read_features(
        ops.join(dataset_dir, 'train_feature.tfrecords'),
        num_epochs=None)
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames],
        batch_size=config.cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2*config.cfg.TRAIN.BATCH_SIZE, #？？？？啥意思
        min_after_dequeue=100,
        num_threads=num_threads)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)#tf.cast：用于改变某个张量的数据类型

    # initialise the net model
    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.cfg.ARCH.HIDDEN_UNITS, # 256
                                     layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
                                     num_classes=len(decoder.char_dict)+1) # 为何+1

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)
    # net_out是啥，[W, N * H, Cls]
    # [width, batch, n_classes]，是一个包含各个字符的概率表
    # TF的ctc_loss:http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
    '''
    net_out: 
            输入（训练）数据，是一个三维float型的数据结构[max_time_step , batch_size , num_classes]

    labels:
            标签序列,是一个稀疏矩阵SparseTensor,由3项组成：http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
               * indices: 二维int64的矩阵，代表非0的坐标点
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
    # CTC的loss，实际上是p(l|x)，l是要探测的字符串，x就是Bi-LSTM输出的x序列
    # 其实就是各个可能的PI的似然概率的和，这个求最大的过程涉及到前向和后向算法了，可以参见CRNN的CTC原理部分
    # 对！对！对！损失函数就是p(l|x)，似然概率之和，使丫最大化。：https://blog.csdn.net/luodongri/article/details/77005948
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels,
                                         inputs=net_out,
                                         sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE)))

    # 这步是在干嘛？是说，你LSTM算出每个时间片的字符分布，然后我用它来做Inference，也就是前向计算
    # 得到一个最大可能的序列，比如"我爱北京天安门"，然后下一步算编辑距离，和标签对比
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)
    # 看，这就是我上面说的编辑距离的差，最小化丫呢
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS,
                                               config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)\
            .minimize(loss=cost, global_step=global_step) #<--- 这个loss是CTC的似然概率值

    # Set tf summary
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)# 这个只是看错的有多离谱，并没有当做损失函数，CTC loss才是核心
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
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
            _, c, seq_distance, preds, gt_labels, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])

            # calculate the precision
            preds = decoder.sparse_tensor_to_str(preds[0])
            gt_labels = decoder.sparse_tensor_to_str(gt_labels)

            accuracy = []

            for index, gt_label in enumerate(gt_labels):
                pred = preds[index]
                total_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            #
            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                    epoch + 1, c, seq_distance, accuracy))

            summary_writer.add_summary(summary=summary, global_step=epoch)
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

    train_shadownet(args.dataset_dir, args.weights_path, args.num_threads)
    print('Done')
