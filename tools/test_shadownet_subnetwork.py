#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-24 下午3:37
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_shadownet_subnetwork.py
# @IDE: PyCharm Community Edition
"""
Test shadow net cnn sub network script
"""
import argparse
import math
import os
import os.path as ops
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from crnn_model import crnn_model
from global_configuration import config
from local_utils import data_utils


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('--weights_path', type=str, help='Where you store the shadow net weights')
    parser.add_argument('--is_recursive', type=bool, help='If need to recursively test the dataset')

    return parser.parse_args()


def test_shadownet_subnetwork(dataset_dir, weights_path, is_vis=False, is_recursive=True):
    """

    :param dataset_dir:
    :param weights_path:
    :param is_vis:
    :param is_recursive:
    :return:
    """
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO().reader
    images_t, labels_t, imagenames_t = decoder.read_features(dataset_dir, num_epochs=None, flag='Test')
    if not is_recursive:
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=32, capacity=1000 + 32 * 2,
                                                                     min_after_dequeue=2, num_threads=4)
    else:
        images_sh, labels_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                                                             batch_size=32, capacity=1000 + 32 * 2, num_threads=4)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25,
                               num_classes=config.cfg.TRAIN.CLASSES_NUMS)

    with tf.variable_scope('shadow'):
        # net_out = net.build_shadownet(inputdata=images_sh)
        net_out = net.build_shadownet_cnn_subnet(inputdata=images_sh)

        preds = tf.argmax(tf.nn.softmax(logits=net_out), axis=1)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0

    test_tfrecords_name = [tmp for tmp in os.listdir(dataset_dir) if
                           re.match(r'^test_feature_\d{0,15}_\d{0,15}\.tfrecords\Z', tmp)]
    for tfrecords_name in test_tfrecords_name:
        for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, tfrecords_name)):
            test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / 32))
    # loops_nums = 100

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if not is_recursive:
            predictions, images, labels, imagenames = sess.run([preds, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            labels = np.argmax(labels, axis=1)

            diff = predictions - labels
            correct_prediction = np.count_nonzero(diff == 0)
            accuracy = correct_prediction / len(labels)

            print('******Image File ID****** ***GT Label*** ***Prediction label***')
            for index, test_image in enumerate(images):
                print('***  Image file {:d}  *** ***  {:d}  *** ***  {:d}  ***'.format(
                    index + 1, labels[index], predictions[index]))
                if is_vis:
                    plt.figure('Test Image')
                    plt.imshow(np.uint8(test_image[:, :, (2, 1, 0)]))
                    plt.show()

            print('Predicted {:d} images {:d} is correct accuracy is {:4f}'.format(
                len(labels), correct_prediction, accuracy))
        else:
            correct_prediction = 0
            test_data_counts = 0
            for loop in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([preds, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                labels = np.argmax(labels, axis=1)

                diff = np.array(predictions - labels)
                correct_prediction += np.count_nonzero(diff == 0)
                test_data_counts += len(labels)

                if loop < 1:
                    print('******File ID****** ***GT Label*** ***Prediction label***')
                for index, test_label in enumerate(labels):
                    print('***  {:d}  *** ***  {:d}  *** ***  {:d}  ***'.format(
                        loop * 32 + index + 1, test_label, predictions[index]))
                    # if is_vis:
                    #     plt.figure('Test Image')
                    #     plt.imshow(images[index][:, :, (2, 1, 0)])
                    #     plt.show()
            accuracy = correct_prediction / test_data_counts

            print('Predicted {:d} images {:d} is correct accuracy is {:4f}'.format(
                test_data_counts, correct_prediction, accuracy))

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test shadow net
    test_shadownet_subnetwork(args.dataset_dir, args.weights_path, args.is_recursive)
