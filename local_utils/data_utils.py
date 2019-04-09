#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午6:46
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_utils.py
# @IDE: PyCharm Community Edition
"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
from typing import List
import numpy as np
import tensorflow as tf
from config import config
from local_utils import log_utils
from local_utils.log_utils import  _p_shape,_p
import re

logger = log_utils.init_logger()

FLAGS = tf.app.flags.FLAGS

## 用来计算预测出来的字符，和label之间的正确率
def caculate_accuracy(preds,labels_sparse,characters):
    # calculate the precision
    preds = sparse_tensor_to_str(preds[0], characters)
    logger.debug("预测结果为：%r", preds)

    # 为何要绕这么一圈，是因为，要通过tensorflow的计算图来读取一遍labels
    labels = sparse_tensor_to_str(labels_sparse, characters)
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


# 把返回的稀硫tensor，转化成对应的字符List
'''
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
def sparse_tensor_to_str( sparse_tensor: tf.SparseTensor,characters) -> List[str]:
    """
    :param sparse_tensor: prediction or ground truth label
    :return: String value of the sparse tensor
    """
    indices = sparse_tensor.indices
    values = sparse_tensor.values #<------------------------ 这个里面存的是string的id，所以要查找字符表，找到对应字符
    values = np.array([characters[id] for id in values])
    dense_shape = sparse_tensor.dense_shape

    # 先初始化一个2维矩阵，用['\n']来填充，因为这个字符不会出现在结果里面，可以当做特殊字符来处理
    # number_lists，实际上是一个dense向量
    number_lists = np.array([['\n'] * dense_shape[1]] * dense_shape[0], dtype=values.dtype)
    res = []

    #先把values，也就是有的值，拷贝到dense向量number_lists中
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]

    # 遍历这个dense的  number_list的每一行，变成一个字符数组
    for one_row in number_lists:
        res.append(''.join(c for c in one_row if c != '\n'))

    return res

def get_charset():
    charset = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    charset = [ch.strip('\n') for ch in charset]
    return charset


def get_file_list(dir):
    from os import listdir
    from os.path import isfile, join
    file_names = ["data/train_set/"+f for f in listdir(dir) if isfile(join(dir, f))]
    # "data/train_set"
    return file_names


def read_labeled_image_list(image_list_file,dict):
    f = open(image_list_file, 'r')

    rex = re.compile(' ')
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

        label = rex.sub('',label)
        labels.append(label)

    logger.info("最终样本标签数量[%d],样本图像数量[%d]",len(labels),len(filenames))

    return filenames, labels

# 这个是在定义操作，注意不是直接的运行，会在session.run后执行
def read_images_from_disk(input_queue,characters):
    # input_queue[0] = _p(input_queue[0],"从磁盘上读取图片和标注")
    image_content = tf.read_file(input_queue[0])

    example = tf.image.decode_png(image_content, channels=3)
    # logger.debug("原始图像shape：%r", example.get_shape().as_list())

    # 第2个参数size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.
    # 对，是Height，Width
    example = tf.image.resize_images(example, config.cfg.ARCH.INPUT_SIZE, method=0)
    labels = input_queue[1]
    # labels = _p_shape(labels, "解析完的labels")
    # example = _p_shape(example, "解析完的图片")
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
            # logger.debug("字符[%s]被替换成[%s]", one, letter)

        # 看是否在里面
        if letter not in dict:
            # logger.error("句子[%s]的字[%s]不属于词表,剔除此样本",sentence,letter)
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


def prepare_image_labels(label_file,characters):

    # 修改了他的加载，讨厌TFRecord方式，直接用文件方式加载
    # 参考：https://saicoco.github.io/tf3/
    # 参考：https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    image_file_names, labels = read_labeled_image_list(label_file, characters)
    logger.debug("读出训练数据：%d条", len(labels))

    # 把图像路径转化成张量
    image_file_names_tensor = tf.convert_to_tensor(image_file_names, dtype=tf.string)
    # 把标签变成词表ID
    labels = convert_to_id(labels, characters)
    labels = expand_array(labels)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
    logger.debug("将标签转化成Tensor")

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

    images, labels = read_images_from_disk(input_queue, characters)
    labels = _to_sparse_tensor(labels)

    # capacity是这个queue的大小，min_after_dequeue出queue里面最少元素
    # 一旦新的进来填充满，还要做一次shuffle，然后再出队，直到剩min_after_dequeue的数量
    # https://blog.csdn.net/ying86615791/article/details/73864381
    images, labels = tf.train.shuffle_batch(
        tensors=[images, labels],
        batch_size=config.cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2 * config.cfg.TRAIN.BATCH_SIZE,  # ？？？？啥意思
        min_after_dequeue=100,
        num_threads=FLAGS.num_threads)

    # 这块，把批次给转给inputdata了，直接就进入图的构建了
    # 并没有一个类似于传统feed_dict的绑定过程
    inputdata = tf.cast(x=images, dtype=tf.float32)  # tf.cast：用于改变某个张量的数据类型

    inputdata = _p_shape(inputdata, "灌入网络之前的数据")

    return inputdata,labels
