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
import cv2,os
from crnn_model import crnn_model
from config import config
from local_utils import log_utils, data_utils
import tensorflow as tf
import numpy as np

logger = log_utils.init_logger()

def initlize_arguments():
    tf.app.flags.DEFINE_string('model_dir', "model/", 'model dir')
    tf.app.flags.DEFINE_boolean('debug', False, 'debug mode')
    tf.app.flags.DEFINE_string('image_path', '', ' data dir')
    tf.app.flags.DEFINE_string('weights_path', None, 'model path')
    tf.app.flags.DEFINE_integer('num_classes', 9, '')

FLAGS = tf.app.flags.FLAGS

def initialize():
    g = tf.Graph()
    with g.as_default():
        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
        saver = tf.train.Saver()
        logger.debug("创建crnn saver")
        sess = tf.Session(config=sess_config,graph=g)

        if FLAGS.weights_path:
            logger.debug("恢复CRNN模型：%s", FLAGS.weights_path)
            saver.restore(sess,FLAGS.weights_path)
        else:
            ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
            logger.debug("最新的CRNN模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
            saver.restore(sess, ckpt)

        # num_classes = len(codec.reader.char_dict) + 1 if num_classes == 0 else num_classes#这个是在读词表，37个类别，没想清楚？？？为何是37个，26个字母+空格不是37个，噢，对了，还有数字0-9
        charset  = data_utils.get_charset()
        logger.info("加载词表，共%d个",len(charset))

    return sess,charset


def recognize(image_path):
    sess,charset = initialize()
    # image读出来是H,W,C
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # cv2.imwrite(image_path + ".resize.png", image)
    pred_result = pred(image,sess,charset)
    logger.info('解析图片%s为：%s',image_path, pred_result)

def pred(image,sess,charset):

    # size要求是(Width,Height)，但是定义的是Height,Width
    size = (config.cfg.ARCH.INPUT_SIZE[1],config.cfg.ARCH.INPUT_SIZE[0])
    # size 的格式是(w,h)，这个和cv2的格式是不一样的，他的是(H,W,C)，看，反的
    logger.debug("首先图像调整尺寸：%r",size)
    image = cv2.resize(image, size)
    image = np.expand_dims(image, axis=0).astype(np.float32) #增加第一个维度，变成了(1,32,100,3),1就是expand_dims的功劳

    logger.debug("定义TF的OP")
    net = crnn_model.ShadowNet(phase='Test',
                               hidden_nums=config.cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=len(charset))
    h, w = config.cfg.ARCH.INPUT_SIZE  # 32,256 H,W
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3], name='input')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    with tf.variable_scope('shadow'):
        net_out = net.build(inputdata=inputdata)
    decodes, prob = tf.nn.ctc_beam_search_decoder(inputs=net_out,
                                               sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(1),
                                               merge_repeated=False)

    with sess.as_default():
        logger.debug("开始预测,输入为：%r", image.shape)
        preds,__prob = sess.run([decodes,prob], feed_dict={inputdata: image})
        logger.debug("预测的结果结果为：%r",  preds)
        logger.debug("预测的结果概率为：%r",  __prob)

        #将结果，从张量变成字符串数组，session.run(arg)arg是啥类型，就ruturn啥类型
        preds = data_utils.sparse_tensor_to_str(preds[0],charset)

    # sess.close()
    return preds[0]


if __name__ == '__main__':

    initlize_arguments()

    if not os.path.exists(FLAGS.image_path):
        raise ValueError('图片[{:s}]找不到'.format(FLAGS.image_path))

    # recognize the image
    recognize(image_path=FLAGS.image_path)

    print("完成")