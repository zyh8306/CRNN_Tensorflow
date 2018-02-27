#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午5:06
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : demo_shadownet_car_plate.py
# @IDE: PyCharm Community Edition
"""
车牌测试
"""
import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import time
import glob
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import log_utils, data_utils

logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Where you store the image',
                        default='data/test_images')
    parser.add_argument('--weights_path', type=str, help='Where you store the weights',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')

    return parser.parse_args()


def recognize(image_path_dir, weights_path, is_vis=False):
    """

    :param image_path_dir:
    :param weights_path:
    :param is_vis:
    :return:
    """
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=15,
                               num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=15 * np.ones(1), merge_repeated=False)

    decoder = data_utils.TextFeatureIO()

    # config tf session
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    image_paths = glob.glob('{:s}/**/*.jpg'.format(image_path_dir), recursive=True)

    accurate_instance = []

    logger.info('---- 图像名称: ---- 真实标签: ---- 预测标签: ----')

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()

        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 32))
            image = np.expand_dims(image, axis=0).astype(np.float32)

            label = ops.split(image_path)[1].split('_')[-2]

            preds = sess.run(decodes, feed_dict={inputdata: image})

            preds = decoder.writer.sparse_tensor_to_str(preds[0])[0]

            if label == preds:
                accurate_instance.append(1)
            else:
                accurate_instance.append(0)

            logger.info('---- {:s} ---- {:s} ---- {:s} ----'.format(
                ops.split(image_path)[1], label, preds
            ))

            if is_vis:
                plt.figure('CRNN Model Demo')
                plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
                plt.show()
        cost_time = time.time() - t_start
    sess.close()
    accuracy = np.mean(accurate_instance, axis=0)
    logger.info('预测实例: {:d}个, 准确率: {:5f}, 耗时: {:5f}'.format(len(accurate_instance), accuracy, cost_time))

    return


if __name__ == '__main__':
    # Inti args
    args = init_args()
    if not ops.exists(args.image_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_dir))

    # recognize the image
    recognize(image_path_dir=args.image_dir, weights_path=args.weights_path)
