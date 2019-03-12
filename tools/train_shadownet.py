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


def get_charset():
    charset = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    charset = [ch.strip('\n') for ch in charset]
    nclass = len(charset)
    return charset,nclass


def get_file_list(dir):
    from os import listdir
    from os.path import isfile, join
    file_names = ["data/train_set/"+f for f in listdir(dir) if isfile(join(dir, f))]
    # "data/train_set"
    return file_names


def read_labeled_image_list(image_list_file,dict):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    # 从文件中读取样本路径和标签值
    # >data/train/21.png )beiji
    # >data/train/22.png 市平谷区金海
    # >data/train/23.png 江中路53
    for line in f:
        # logger.debug("line=%s",line)
        # filename, label = line[:-1].split(' ')
        filename , _ , label = line[:-1].partition(' ') # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车

        label = process_unknown_charactors(label,dict)

        # 如果此样本属于剔除样本，忽略之
        if label is None: continue

        filenames.append(filename)
        labels.append(label)

    logger.info("最终样本标签数量[%d],样本图像数量[%d]",len(labels),len(filenames))

    return filenames, labels


def read_images_from_disk(input_queue,characters):

    image_content = tf.read_file(input_queue[0])
    example = tf.image.decode_png(image_content, channels=3)
    logger.debug("原始图像shape：%r", example.get_shape().as_list())
    example = tf.image.resize_images(example, config.cfg.ARCH.INPUT_SIZE, method=0)
    logger.debug("Resized图像shape：%r", example.get_shape().as_list())
    labels = input_queue[1]

    return example, labels

def process_unknown_charactors(sentence,dict):
    unkowns = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／"
    knows = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/"

    result = ""
    for one in sentence:
        # 对一些特殊字符进行替换，替换成词表的词
        i = unkowns.find(one)
        if i==-1:
            letter = one
        else:
            letter = knows[i]
            logger.debug("字符[%s]被替换成[%s]", one, letter)

        # 看是否在里面
        if letter not in dict:
            logger.error("句子[%s]的字[%s]不属于词表,剔除此样本",sentence,letter)
            return None

        result+= letter
    return result





# labels是所有的标签的数组['我爱北京','我爱天安门',...,'他说的法定']
# characters:词表
def convert_to_id(labels,characters):

    _lables = []
    for one in labels:
        _lables.append( [characters.index(l) for l in one] )

    return _lables

# 原文：https://blog.csdn.net/he_wen_jie/article/details/80586345
# 入参：
# sequence
# [
#   [123,44,22],
#   [23,44,55,4445,334,453],
#   ..
# ]
def to_sparse_tensor(sequences, dtype=np.int32):
    indices = [] # 位置,哪些位置上非0
    values = [] # 具体的值

    for n, seq in enumerate(sequences): # sequences是一个二维list
        indices.extend(zip([n]*len(seq), range(len(seq)))) # 生成所有值的坐标，不管是不是0，都存下来
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences),np.asarray(indices).max(0)[1]+1],
                       dtype=np.int32) # shape的行就是seqs的个数，列就是最长的那个seq的长度
    logger.debug("labels被转化的sparse的tensor的shape:%r", shape)
    return tf.SparseTensor(indices, values, shape)


def expand_array(data):
    max = 0
    for one in data:
        if len(one)>max:
            max = len(one)

    for one in data:
        one.extend( [0] * (max - len(one)) )

    return data

def _to_sparse_tensor(dense):
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)
    return sparse
    #labels_tensor = to_sparse_tensor(labels)  # 把label从id数组，变成张量


def train_shadownet(dataset_dir, weights_path=None, num_threads=4):

    #为了兼容下面的代码，先留着
    decoder = data_utils.TextFeatureIO().reader

    # 2.26 piginzoo 之前的tfrecord方式，不爽，改了
    # """
    # :param dataset_dir:
    # :param weights_path:
    # :param num_threads: Number of threads to use in tf.train.shuffle_batch
    # :return:
    # """
    # # decode the tf records to get the training data
    # decoder = data_utils.TextFeatureIO().reader
    # #使用的是tensorflow的tfrecords格式，预加载文件形式，直接load到内存，速度快
    # images, labels, imagenames = decoder.read_features(
    #     ops.join(dataset_dir, 'train_feature.tfrecords'),
    #     num_epochs=None)
    #
    # # shuffle_batch:该方法可以对输入的tensors生成对应的每个batch大小为batch_size的队列
    # # 他返回的是张量：返回列表或者字典，类型为tensor，形状数量与tensors_list中元素大小一致
    # inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
    #     tensors=[images, labels, imagenames],
    #     batch_size=config.cfg.TRAIN.BATCH_SIZE,
    #     capacity=1000 + 2*config.cfg.TRAIN.BATCH_SIZE, #？？？？啥意思
    #     min_after_dequeue=100,
    #     num_threads=num_threads)

    characters, num_classes = get_charset()

    # 修改了他的加载，讨厌TFRecord方式，直接用文件方式加载
    # 参考：https://saicoco.github.io/tf3/
    # 参考：https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

    image_file_names, labels = read_labeled_image_list("data/train.txt",characters)
    # logger.debug("读出")
    # logger.debug("image_file_names")
    # logger.debug(image_file_names)
    # logger.debug("lables")
    # logger.debug(labels)

    # 把图像路径转化成张量
    image_file_names_tensor = tf.convert_to_tensor(image_file_names, dtype=tf.string)
    # 把标签变成词表ID
    labels = convert_to_id(labels, characters)
    labels = expand_array(labels)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    # 我尝试了多种思路：
    # 1.作者原来的思路：
    #   原来的是把label+image，一口气都写入TFRecord，这样就是相当于绑定了2者，然后，用tf.train.string_input_producer，产生epochs
    #   就是把文件重复读几次，不过我怀疑，他也是一次性载入，担心内存。。。（不过，怀疑归怀疑，怎么验证呢）
    #   然后用tf.parse_single_example还原到内存里，然后调用tf.train.shuffle_batch形成批次，
    #   tf.train.shuffle_batch里面明显有个queue，应该就是存放tffeature的，这个时候我理解又成了一条一条的，
    #   和我之前认为加载了整个文件相矛盾，所以，我更愿意相信，他是一条条加载的，这样节省内存。
    # 2. 可以我偏不，我不喜欢先写成一个大文件，然后再读，于是，我尝试自己来做。
    #   啥叫自己来做，就是自己来控制批次，其实，我们主要就是干两件事，一个是控制epochs，一个是控制batch
    #   我看到一种做法是遍历，for epochs; for batch {...}，也就是在{}里面去做sess.run，通过feed_dict把这个批次传入
    #   这样做没都做一次梯度下降，简单易懂，挺好的。没用到啥tf.train.shuffle_batch，也没用到tf.train.start_queue_runners/tf.train.Coordinator().
    #   可是，我要就着作者之前的代码，用tf.train.Coordinator()+tf.train.start_queue_runners()+tf.train.shuffle_batch()的方式来干。
    #   然后，就No作No逮了，我遇到了一系列问题：
    #       - 要用slice_input_producer加载文件和标签了，生成多个epochs了，不能用之前string_input_producer的方式来生成epochs了，
    #         string_是用来生成单个文件名的队列的，我现在的要的是整个image_files+labels(所有的）生成多个epochs
    #       - 我要转成Tensorflow的tf.nn.ctc_loss输入所需要的SparseTensor:labels
    #   可是，问题出现了，就是在slice_input_producer的时候就卡住了，我先做了labels=>SparseTensor的转化，
    #   然后调用slice_input_producer的时候报错：
    #   TypeError: Failed to convert object of type <class 'tensorflow.python.framework.sparse_tensor.SparseTensor'> to Tensor.
    #   我的解决办法是，
    #
    #
    #
    #   另外，这种的方法，我还有一个顾虑，因为即使可以这样做，每一次都要做一个image_names+labels的tensor转化，
    #   转化的时候，实际上是把这些数据，保存到了Graph里，也就是内存里，如果是100万+的数据，这个量也很大了，这也是个潜在的问题，
    #   不知道TFRecord的方式是如何避免了这个问题，他真的能做到只加载部分的数据么？唉，还是喜欢那种简单直白的方式。
    #   吐槽一下：张量的方式真蛋疼，只定义操作，不涉及数据，数据都是后期绑定的。简单的方法是靠feed_dict，复杂的就是靠start_queue_runners()+Coordinator了
    #

    # https://stackoverflow.com/questions/48201725/converting-tensor-to-a-sparsetensor-for-ctc-loss


    # 我办法用string_input_producer，因为它只能用来来支持一维的文件名，往往是用来加载文件名队列的
    # 我的需求是，既有文件名，又有label，所以，只能用slice_input_producer，他支持list，
    # 这样，我就可以把label+image的组合，做一个加载了，加载多少次当然是由num_epochs来决定的
    input_queue = tf.train.slice_input_producer([image_file_names_tensor, labels_tensor],
                                                num_epochs=config.cfg.TRAIN.EPOCHS,
                                                shuffle=True)

    images, labels = read_images_from_disk(input_queue,characters)
    labels = _to_sparse_tensor(labels)

    # capacity是这个queue的大小，min_after_dequeue出queue里面最少元素
    # 一旦新的进来填充满，还要做一次shuffle，然后再出队，直到剩min_after_dequeue的数量
    # https://blog.csdn.net/ying86615791/article/details/73864381
    images, labels = tf.train.shuffle_batch(
        tensors=[images, labels],
        batch_size=config.cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2 * config.cfg.TRAIN.BATCH_SIZE,  # ？？？？啥意思
        min_after_dequeue=100,
        num_threads=num_threads)

    # 这块，把批次给转给inputdata了，直接就进入图的构建了
    # 并没有一个类似于传统feed_dict的绑定过程
    inputdata = tf.cast(x=images, dtype=tf.float32) # tf.cast：用于改变某个张量的数据类型

    # initialise the net model
    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.cfg.ARCH.HIDDEN_UNITS, # 256
                                     layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
                                     num_classes=num_classes)

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
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=labels,
                                         inputs=net_out,
                                         sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE)))

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
    # 看，这就是我上面说的编辑距离的差，最小化丫呢
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

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
    if not os.path.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
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
            # session.run(): 第一个参数fetches: The fetches argument may be a single graph element,
            # or an arbitrarily nested list, tuple, namedtuple, dict,
            # or OrderedDict containing graph elements at its leaves.
            _, c, seq_distance, preds, gt_labels, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, labels, merge_summary_op])
            # labels<----标签张量，应该是分批次的把？

            # calculate the precision
            # logger.debug("预测出来的结果为：%r",preds[0])
            preds = decoder.sparse_tensor_to_str(preds[0],characters)
            gt_labels = decoder.sparse_tensor_to_str(gt_labels,characters)

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

    if not os.path.exists(args.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))

    train_shadownet(args.dataset_dir, args.weights_path, args.num_threads)
    print('Done')
