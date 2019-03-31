#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : demo_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text
"""
import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from config import config
from local_utils import log_utils, data_utils

logger = log_utils.init_logger()

tf.app.flags.DEFINE_boolean('debug', False, 'debug mode')
tf.app.flags.DEFINE_string('image_path', '', ' data dir')
tf.app.flags.DEFINE_string('weights_path', None, 'model path')
tf.app.flags.DEFINE_integer('num_classes', 9, '')
FLAGS = tf.app.flags.FLAGS


def recognize(image_path: str, weights_path: str, is_vis: bool=True, num_classes: int=0):
    """

    :param image_path:
    :param weights_path:
    :param is_vis:
    :param num_classes:
    """

    # image读出来是H,W,C
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # size要求是(Width,Height)，但是定义的是Height,Width
    size = (config.cfg.ARCH.INPUT_SIZE[1],config.cfg.ARCH.INPUT_SIZE[0])

    # size 的格式是(w,h)，这个和cv2的格式是不一样的，他的是(H,W,C)，看，反的
    image = cv2.resize(image, size)

    cv2.imwrite(image_path+".resize.png",image)

    image = np.expand_dims(image, axis=0).astype(np.float32) #增加第一个维度，变成了(1,32,100,3),1就是expand_dims的功劳

    h,w = config.cfg.ARCH.INPUT_SIZE #32,256 H,W
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3], name='input')

    # num_classes = len(codec.reader.char_dict) + 1 if num_classes == 0 else num_classes#这个是在读词表，37个类别，没想清楚？？？为何是37个，26个字母+空格不是37个，噢，对了，还有数字0-9
    characters , num_classes = data_utils.get_charset()

    net = crnn_model.ShadowNet(phase='Test',
                               hidden_nums=config.cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=num_classes)

    with tf.variable_scope('shadow'):
        net_out = net.build(inputdata=inputdata)

    decodes, prob = tf.nn.ctc_beam_search_decoder(inputs=net_out,
                                               sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(1),
                                               merge_repeated=False)

    logger.debug("CTC后的结果：")
    logger.debug("decode:%r",decodes)
    logger.debug("prob:%r",prob)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        logger.debug("恢复模型：%s",weights_path)

        logger.debug("预测的输入为：%r", image.shape)
        preds,__prob = sess.run([decodes,prob], feed_dict={inputdata: image})
        logger.debug("预测的结果preds为：%r",  preds)
        logger.debug("预测的结果prob为：%r",  __prob)

        #将结果，从张量变成字符串数组，session.run(arg)arg是啥类型，就ruturn啥类型
        preds = data_utils.sparse_tensor_to_str(preds[0],characters)

        logger.info('解析图片{:s}为：{:s}'.format(ops.split(image_path)[1], preds[0]))

        # if is_vis:
        #     plt.figure('CRNN Model Demo')
        #     plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
        #     plt.show()

        sess.close()


if __name__ == '__main__':

    if not ops.exists(FLAGS.image_path):
        raise ValueError('{:s} doesn\'t exist'.format(FLAGS.image_path))

    # recognize the image
    recognize(image_path=FLAGS.image_path, weights_path=FLAGS.weights_path, num_classes=FLAGS.num_classes)

    print("Done")