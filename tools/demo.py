#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : demo.py
# @IDE: PyCharm Community Edition
"""
Use CRNN net to recognize the scene text
"""
import logging
import tensorflow as tf
import os.path as ops
import numpy as np
import argparse
from crnn_model import model
from config import config
from utils import image_util
from utils import log_util
from utils import text_util
from utils import tensor_util


# 初始化日志
log_util.init_logging()
logger = logging.getLogger()


def init_args() -> argparse.Namespace:
    """
    初始化参数解析
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_name', type=str, help='Where you store the image',
                        default='6.png')
    parser.add_argument('-w', '--weights_dir', type=str, help='the directory you store the weights',
                        default='model/')
    parser.add_argument('-c', '--charset_file', type=str, help='where charset file is',
                        default='charset6k.txt')
    return parser.parse_args()


def recognize(image_path: str, weights_dir: str, charset_file: str):
    """
    开始识别单张图片
    :param image_path:
    :param weights_dir:
    :param charset_file:
    """

    # 获取字符库
    characters = text_util.get_charset(charset_file)

    # 获取图片
    image = image_util.read_image_file(image_path)
    image_list = image_util.resize_batch_image([image], 'RESIZE_FORCE', config.cfg.ARCH.INPUT_SIZE)

    # 获取宽高
    image = image_list[0]
    height = image.shape[0]
    width = image.shape[1]
    sequence_len = width//4

    # 定义张量
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, height, width, 3], name='input')

    net = model.CrnnNet(
            phase='Test',
            hidden_num=config.cfg.ARCH.HIDDEN_UNITS, # 256
            layers_num=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
            num_classes=len(characters) + 1)

    with tf.variable_scope('shadow'):
        net_out = net.build_network(images=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=sequence_len*np.ones(1),
                                               merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(weights_dir)

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=save_path)

        preds = sess.run(decodes, feed_dict={inputdata: image_list})

        texts = tensor_util.sparse_tensor_to_str(preds[0], characters)

        logger.info('Image[ {:s} ] Predict ==> {:s}'.format(ops.split(image_path)[1], texts[0]))

        sess.close()


if __name__ == '__main__':
    args = init_args()

    image_path = "data/train/" + args.image_name
    if not ops.exists(image_path):
        raise ValueError('{:s} doesn\'t exist'.format(image_path))

    # recognize the image
    recognize(image_path=image_path, weights_dir=args.weights_dir, charset_file=args.charset_file)
