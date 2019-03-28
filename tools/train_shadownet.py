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
import time
import numpy as np
from local_utils.log_utils import  _p_shape
from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config


tf.app.flags.DEFINE_boolean('debug', False, 'debug mode')
tf.app.flags.DEFINE_string('dataset_dir', '', 'train data dir')
tf.app.flags.DEFINE_string('weights_path', None, 'model path')
tf.app.flags.DEFINE_integer('validate_steps', 10, 'model path')
tf.app.flags.DEFINE_integer('num_threads', 4, 'read train data threads')


FLAGS = tf.app.flags.FLAGS


logger = log_utils.init_logger()


def caculate_accuracy(preds,labels_sparse,characters):
    # calculate the precision
    logger.debug("参数更新后，预测出来的结果为：%r", preds)
    preds = data_utils.sparse_tensor_to_str(preds[0], characters)
    # 为何要绕这么一圈，是因为，要通过tensorflow的计算图来读取一遍labels
    labels = data_utils.sparse_tensor_to_str(labels_sparse, characters)

    accuracy = []

    for index, _label in enumerate(labels):
        pred = preds[index]
        total_count = len(_label)
        correct_count = 0
        try:
            for i, tmp in enumerate(_label):
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
    return accuracy

'''
CRNN的训练epoch耗时：
3.5分钟1个epochs，10万次样本，3000个batches
35分钟10个epochs，100万次样本
350分钟100个epochs  5.8小时，1000万次样本
3500分钟，1000个epochs 58小时，1亿次样本
'''
def train_shadownet(dataset_dir, weights_path=None, num_threads=4):

    logger.info("开始训练")

    characters, num_classes = data_utils.get_charset()

    # 注意噢，这俩都是张量
    train_images_tensor,train_labels_tensor = data_utils.prepare_image_labels(characters)

    # initialise the net model
    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.cfg.ARCH.HIDDEN_UNITS, # 256
                                     layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
                                     num_classes=num_classes)

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=train_images_tensor)

    logger.debug("网络构建完毕")

    # net_out是啥，[W, N * H, Cls]
    # [width, batch, n_classes]，是一个包含各个字符的概率表
    # TF的ctc_loss:http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
    '''
    net_out: 
            输入（训练）数据，是一个三维float型的数据结构[max_time_step , batch_size , num_classes]

    labels:
            标签序列,是一个稀疏矩阵SparseTensor,由3项组成：http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
               * indices: 二维int32的矩阵，代表非0的坐标点
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
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=train_labels_tensor,
                                         inputs=net_out,
                                         sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE)))

    logger.debug("cost损失函数构建完毕")

    # 这步是在干嘛？是说，你LSTM算出每个时间片的字符分布，然后我用它来做Inference，也就是前向计算
    # 得到一个最大可能的序列，比如"我爱北京天安门"，然后下一步算编辑距离，和标签对比
    # ？？？ B变化，也就是去掉空格和重复的过程在这里面做了么？
    # 答：没有做，原因是：merge_repeated=False
    # 原因：摘自tf.nn.ctc_beam_search_decoder()
    #       If `merge_repeated` is `True`, merge repeated classes in the output beams.
    #       This means that if consecutive entries in a beam are the same,
    #       only the first of these is emitted.  That is, when the sequence is
    #       `A B B * B * B` (where '*' is the blank label), the return value is:
    #          * `A B` if `merge_repeated = True`.
    #          * `A B B B` if `merge_repeated = False`.
    #
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)
    logger.debug("CTC网络构建完毕")

    # 看，这就是我上面说的编辑距离的差，最小化丫呢
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), train_labels_tensor))

    global_step = tf.Variable(0, name='global_step', trainable=False)


    learning_rate = tf.train.exponential_decay(config.cfg.TRAIN.LEARNING_RATE,
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
    if not os.path.exists(tboard_save_path): os.makedirs(tboard_save_path)

    accuracy = tf.Variable(0, name='accuracy', trainable=False)
    tf.summary.scalar(name='Accuracy', tensor=accuracy)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)# 这个只是看错的有多离谱，并没有当做损失函数，CTC loss才是核心

    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)
    logger.debug("创建session")

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = config.cfg.TRAIN.EPOCHS

    with sess.as_default():

        sess.run(tf.local_variables_initializer())
        if weights_path is None:
            logger.info('从头开始训练，不加载旧模型')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('从文件{:s}恢复模型，继续训练'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator() # 创建一个协调器：http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 哦，协调器，不用关系数据的批量获取，他只是一个线程和Queue操作模型，数据的获取动作是由shuffle_batch来搞定的
        # 只不过搞定这事是在不同线程和队列里完成的

        for epoch in range(train_epochs):
            logger.debug("第%d次训练",epoch)
            # session.run(): 第一个参数fetches: The fetches argument may be a single graph element,
            # or an arbitrarily nested list, tuple, namedtuple, dict,
            # or OrderedDict containing graph elements at its leaves.
            _, c, seq_distance, preds, labels_sparse_tensor, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, train_labels_tensor, merge_summary_op])

            # 每个一定步骤（目前设置是10），就计算一下编辑距离，并且验证一下
            if epoch % FLAGS.validate_steps == 0:
                _, c, seq_distance, preds, labels_sparse, summary = sess.run(
                    [optimizer, cost, sequence_dist, decoded, train_labels_tensor, merge_summary_op])
                _accuracy = caculate_accuracy(preds, labels_sparse)
                tf.assign(accuracy, _accuracy) # 更新正确率变量
                logger.info('Epoch: {:d} Train accuracy= {:9f}'.format(epoch + 1, _accuracy))
            else:
                _, ctc_lost, summary = sess.run([optimizer, cost, merge_summary_op])


            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f}'.format(epoch + 1, c, seq_distance))

            summary_writer.add_summary(summary=summary, global_step=epoch)

            # 10万个样本，一个epoch是3.5分钟，CHECKPOINT_STEP=20，大约是70分钟存一次
            if epoch % config.cfg.TRAIN.CHECKPOINT_STEP == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return




if __name__ == '__main__':

    if not os.path.exists(FLAGS.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(FLAGS.dataset_dir))

    print("开始训练")
    train_shadownet(FLAGS.dataset_dir, FLAGS.weights_path, FLAGS.num_threads)

