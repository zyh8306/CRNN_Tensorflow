#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Script
"""
import os
import tensorflow as tf
import time
import numpy as np
import datetime
from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from config import config
from local_utils.log_utils import _p_shape

tf.app.flags.DEFINE_string('name', 'CRNN', 'no use ,just a flag for shell batch')
tf.app.flags.DEFINE_boolean('debug', False, 'debug mode')
tf.app.flags.DEFINE_string('train_dir','data/train','')
tf.app.flags.DEFINE_string('label_file','train.txt','')
tf.app.flags.DEFINE_string('tboard_dir', 'tboard', 'tboard data dir')
tf.app.flags.DEFINE_string('weights_path', None, 'model path')
tf.app.flags.DEFINE_integer('validate_steps', 10000, 'model path')
tf.app.flags.DEFINE_integer('num_threads', 4, 'read train data threads')
FLAGS = tf.app.flags.FLAGS

logger = log_utils.init_logger()


def caculate_accuracy(preds,labels_sparse,characters):
    # calculate the precision
    preds = data_utils.sparse_tensor_to_str(preds[0], characters)
    logger.debug("预测结果为：%r", preds)

    # 为何要绕这么一圈，是因为，要通过tensorflow的计算图来读取一遍labels
    labels = data_utils.sparse_tensor_to_str(labels_sparse, characters)
    logger.debug("标签为：%r", labels)

    accuracy = []

    # 挨个遍历标签，
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


def save_model(saver,sess,epoch):
    model_save_dir = 'model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)
    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    logger.info("训练: 保存了模型：%s", model_save_path)


def create_summary_writer(sess):
    # 按照日期，一天生成一个Summary/Tboard数据目录
    # Set tf summary
    if not os.path.exists(FLAGS.tboard_dir): os.makedirs(FLAGS.tboard_dir)
    today = datetime.datetime.now().strftime("%Y%m%d")
    summary_dir = os.path.join(FLAGS.tboard_dir,today)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_writer.add_graph(sess.graph)
    return summary_writer


def train(weights_path=None):
    '''
    CRNN的训练epoch耗时：
    3.5分钟1个epochs，10万次样本，3000个batches
    35分钟10个epochs，100万次样本
    350分钟100个epochs  5.8小时，1000万次样本
    3500分钟，1000个epochs 58小时，1亿次样本
    '''

    logger.info("开始训练")

    characters, num_classes = data_utils.get_charset()

    # 注意噢，这俩都是张量
    train_images_tensor,train_labels_tensor = \
        data_utils.prepare_image_labels(FLAGS.label_file,characters)

    # 创建模型
    network = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.cfg.ARCH.HIDDEN_UNITS, # 256
                                     layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
                                     num_classes=num_classes)
    with tf.variable_scope('shadow', reuse=False):
        net_out = network.build(inputdata=train_images_tensor)

    # 创建优化器和损失函数的op
    cost,optimizer,global_step = network.loss(net_out,train_labels_tensor)

    # 创建校验用的decode和编辑距离
    decode, sequence_dist = network.validate(net_out,train_labels_tensor)

    # 创建一个变量用于把计算的精确度加载到summary中
    accuracy = tf.Variable(0, name='accuracy', trainable=False)
    tf.summary.scalar(name='validate.Accuracy', tensor=accuracy)

    train_summary_op = tf.summary.merge_all(scope="train")
    validate_summary_op = tf.summary.merge_all(scope="validate")
    _p_shape(train_summary_op,"训练阶段的Summary收集")
    _p_shape(train_summary_op,"校验阶段的Summary收集")

    # Set saver configuration
    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)
    logger.debug("创建session")

    summary_writer = create_summary_writer(sess)

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
            global_step = log_utils._p(global_step,"加载模型的时候，得到的global_step")
            tf.assign(global_step,0)

        coord = tf.train.Coordinator() # 创建一个协调器：http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 哦，协调器，不用关系数据的批量获取，他只是一个线程和Queue操作模型，数据的获取动作是由shuffle_batch来搞定的
        # 只不过搞定这事是在不同线程和队列里完成的

        for epoch in range(train_epochs):
            logger.debug("训练: 第%d次",epoch)

            # 每个一定步骤（目前设置是10），就计算一下编辑距离，并且验证一下
            if epoch!=0 and epoch % FLAGS.validate_steps == 0:

                # 梯度下降，并且采集各种数据：编辑距离、预测结果、输入结果、训练summary和校验summary
                # 这过程非常慢，32batch的实测在K40的显卡上，实测需要15分钟
                _,          seq_distance,  preds,  labels_sparse,      v_summary,          t_summary = sess.run(
                [optimizer, sequence_dist, decode, train_labels_tensor,validate_summary_op,train_summary_op])
                logger.info('训练: Epoch: {:d}训练结束'.format(epoch + 1))

                _accuracy = caculate_accuracy(preds, labels_sparse,characters)
                tf.assign(accuracy, _accuracy) # 更新正确率变量
                logger.info('正确率计算完毕：%f', _accuracy)

                summary_writer.add_summary(summary=v_summary, global_step=epoch)
                logger.debug("写入校验、距离计算、正确率Summary")
                summary_writer.add_summary(summary=t_summary, global_step=epoch)
                logger.debug("写入训练Summary")


            else:
                # 单纯训练
                _, ctc_lost, t_summary = sess.run([optimizer, cost, train_summary_op])
                logger.debug("训练: 优化完成、cost计算完成、Summary写入完成")
                summary_writer.add_summary(summary=t_summary, global_step=epoch)
                logger.debug("写入训练Summary")


            # 10万个样本，一个epoch是3.5分钟，CHECKPOINT_STEP=20，大约是70分钟存一次
            if epoch % config.cfg.TRAIN.CHECKPOINT_STEP == 0:
                save_model(saver,sess,epoch)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return




if __name__ == '__main__':

    print("开始训练...")
    train(FLAGS.weights_path)

