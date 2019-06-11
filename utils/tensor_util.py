import datetime

import numpy as np
import tensorflow as tf
import logging
import re,cv2

logger = logging.getLogger("data factory")


## 用来计算预测出来的字符，和label之间的正确率
def calculate_accuracy(preds, labels_sparse, characters):
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


def sparse_tensor_to_str( sparse_tensor: tf.SparseTensor, characters):
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
    return [indices, values, shape]


def convert_to_sparse_tensor(dense):
    zero = tf.constant(-1, dtype=tf.int32)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)
    return sparse


# 打印tensor的值
def print_tensor(tensor, msg, debug=False):
    if debug:
        dt = datetime.datetime.now().strftime('TF_PRINT: %m-%d %H:%M:%S: ')
        msg = dt + msg
        return tf.Print(tensor, [tensor], msg, summarize=100)
    else:
        return tensor


# 打印tensor的shape
def print_tensor_shape(tensor, msg, debug=False):
    if debug:
        dt = datetime.datetime.now().strftime('TF_PRINT: %m-%d %H:%M:%S: ')
        msg = dt + msg
        return tf.Print(tensor, [tf.shape(tensor)], msg, summarize=100)
    else:
        return tensor

