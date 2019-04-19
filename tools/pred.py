#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : Lu
# @Site    : http://github.com/TJCVRS
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

FLAGS = tf.app.flags.FLAGS

# 初始化3任务：1、构建图 2、加载model=>session 3、加载字符集，都放到全局变量里
def initialize():

    # 为了在不同的web请求间共享信息，需要把这些张量变量共享成全局变量
    global charset, decodes, prob, inputdata,batch_size

    g = tf.Graph()
    with g.as_default():
        # num_classes = len(codec.reader.char_dict) + 1 if num_classes == 0 else num_classes#这个是在读词表，37个类别，没想清楚？？？为何是37个，26个字母+空格不是37个，噢，对了，还有数字0-9
        charset  = data_utils.get_charset(FLAGS.charset)
        logger.info("加载词表，共%d个",len(charset))

        decodes, prob, inputdata, batch_size = build_graph(g,charset)

        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
        saver = tf.train.Saver()
        logger.debug("创建crnn saver")
        sess = tf.Session(config=sess_config,graph=g)

        if FLAGS.crnn_model_file:
            crnn_model_file_path = os.path.join(FLAGS.crnn_model_dir,FLAGS.crnn_model_file)
            logger.debug("恢复给定名字的CRNN模型：%s", crnn_model_file_path)
            saver.restore(sess,crnn_model_file_path)
        else:
            ckpt = tf.train.latest_checkpoint(FLAGS.crnn_model_dir)
            logger.debug("最新CRNN模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
            saver.restore(sess, ckpt)

    return sess


def recognize():

    image_list = []
    if (FLAGS.image_file):
        image_path = os.path.join(FLAGS.image_dir,FLAGS.image_file)
        image = cv2.imread(image_path)
        image_list.append(image)
        logger.debug("加载了图片:%s",image_path)
    else:
        # 遍历图片目录
        logger.debug("加载目录[%s]下的所有图片",FLAGS.image_dir)
        image_names = os.listdir(FLAGS.image_dir)
        for image_name in image_names:

            _,subfix = os.path.splitext(image_name)
            if subfix.lower() not in ['.jpg','.png','.jpeg','.gif','.bmp']: continue

            image_path = os.path.join(FLAGS.image_dir, image_name)
            image = cv2.imread(image_path)
            logger.debug("加载图片:%s", image_path)
            image_list.append(image)

    sess = initialize()
    # batch_size，也就是CTC的sequence_length数组要求的格式是：
    # 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
    _batch_size = np.array(len(image_list)*[config.cfg.ARCH.SEQ_LENGTH]).astype(np.int32)

    pred_result = pred(image_list,_batch_size,sess)

    logger.info('解析图片%s为：%s',image_path, pred_result)


def build_graph(g,charset):
    with g.as_default():
        logger.debug("定义TF计算图")
        net = crnn_model.ShadowNet(phase='Test',
                                   hidden_nums=config.cfg.ARCH.HIDDEN_UNITS,
                                   layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,
                                   num_classes=len(charset))
        h, w = config.cfg.ARCH.INPUT_SIZE  # 32,256 H,W
        inputdata = tf.placeholder(dtype=tf.float32,
                                   shape=[None, h, w, 3],
                                   name='input')
        # 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
        # 这里不定义，当做placeholder，后期session.run时候传入
        batch_size = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope('shadow'):
            net_out = net.build(inputdata=inputdata,sequence_len=batch_size)

        net_out =log_utils._p_shape(net_out,"LSTM运行态的输出")
        logger.debug("CTC输入网络的维度为：%r",net_out.get_shape().as_list())

        # inputs: 3-D tensor,shape[max_time x batch_size x num_classes]
        decodes, prob = tf.nn.ctc_beam_search_decoder(inputs=net_out,
                                                      beam_width=5,
                                                      sequence_length = batch_size,
                                                      merge_repeated=False)
                                                      #sequence_length=np.array(batch*[config.cfg.ARCH.SEQ_LENGTH]),

        return decodes,prob,inputdata,batch_size

# 把传入的图片数组(opencv BGR格式的)转成神经网络的张量的样子=>[Batch,H,W,C]
def prepare_data(image_list):
    input_data = []
    for image in image_list:
        # size要求是(Width,Height)，但是定义的是Height,Width
        size = (config.cfg.ARCH.INPUT_SIZE[1],config.cfg.ARCH.INPUT_SIZE[0])
        # size 的格式是(w,h)，这个和cv2的格式是不一样的，他的是(H,W,C)，看，反的
        # image = cv2.resize(image, size) 2019.4.19 piginzoo，resize改成padding操作
        image = data_utils.padding(image)
        image = image[:,:,::-1] # 每张图片要BGR=>RGB顺序
        input_data.append(image)
    return np.array(input_data)


# 输入是图像numpy数据，注意，顺序是RGB，注意OpenCV read的数据是BGR，要提前转化后再传给我
def pred(image_list,_batch_size,sess):
    global charset, decodes, prob, inputdata, batch_size

    result = []
    for i in range(0,len(image_list),_batch_size):
        batch_num = _batch_size
        if i+_batch_size> len(image_list):
            batch_num = len(image_list)

        _input_data = image_list[i:batch_num]

        _input_data = prepare_data(_input_data)

        with sess.as_default():
            logger.debug("开始预测,输入的数据为：%r", _input_data.shape)
            preds,__prob = sess.run(
                [decodes,prob],
                feed_dict={
                    inputdata :_input_data,
                    batch_size:_batch_size
                })
            # 将结果，从张量变成字符串数组，session.run(arg)arg是啥类型，就ruturn啥类型
            preds = data_utils.sparse_tensor_to_str(preds[0],charset)
            logger.debug("预测的结果结果为：%r",  preds)
            logger.debug("预测的结果概率为：%r",  __prob)
            result+= preds

    return result

# 如果不指定文件名，识别data/test/目录下的所有图片，否则具体的照片
if __name__ == '__main__':

    tf.app.flags.DEFINE_string('crnn_model_dir', None,'')
    tf.app.flags.DEFINE_string('crnn_model_file', None,'')
    tf.app.flags.DEFINE_boolean('debug', True,'')
    tf.app.flags.DEFINE_string('charset','', '')
    tf.app.flags.DEFINE_string('image_dir', None,'')
    tf.app.flags.DEFINE_string('image_file', None,'')
    tf.app.flags.DEFINE_integer('num_classes', 5990,'')

    if not os.path.exists(FLAGS.charset):
        logger.error("字符集文件[%s]不存在",FLAGS.charset)
        exit()
    if not os.path.exists(FLAGS.image_dir):
        logger.error("要识别的图片的目录[%s]不存在",FLAGS.image_dir)
        exit()
    if FLAGS.image_file and not os.path.exists(os.path.join(FLAGS.image_dir,FLAGS.image_file)):
        logger.error("要识别的图片[%s]不存在",os.path.join(FLAGS.image_dir,FLAGS.image_file))
        exit()
    if not os.path.exists(FLAGS.crnn_model_dir):
        logger.error("模型目录[%s]不存在",FLAGS.crnn_model_dir)
        exit()
    if FLAGS.crnn_model_file and not os.path.exists(os.path.join(FLAGS.crnn_model_dir,FLAGS.crnn_model_file+".meta")):
        logger.error("模型文件[%s]不存在",os.path.join(FLAGS.crnn_model_dir,FLAGS.crnn_model_file,".meta"))
        exit()

    # 识别
    recognize()
    print("完成")

    # 测试prepare_data()
    # p0 = cv2.imread("test/0.png")
    # p1 = cv2.imread("test/1.png")
    # image_list = [p0,p1]
    # data = prepare_data(image_list)
    # print(data.shape)