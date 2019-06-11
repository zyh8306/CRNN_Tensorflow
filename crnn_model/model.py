"""
Implement for the crnn model mentioned in "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition"

https://arxiv.org/abs/1507.05717v1
"""

import logging
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import slim

from utils import tensor_util

_BATCH_DECAY = 0.999
logger = logging.getLogger("crnn")


class CrnnNet(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes):
        """
        CrnnNet构造函数
        :param phase:
        :param hidden_num: 隐藏单元个数，这里是256
        :param layers_num: 隐藏层个数，这里是2层
        :param num_classes: 识别字符数量，这里需要注意 num_classes = 实际字符数 + 1 。
                             +1 是为了应对CTC的 blank ，所有ctc相关函数规定，num_classes的最后一位就是blank
        :return:
        """
        self.__phase = phase.lower()
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        self.__num_classes = num_classes
        return

    def __feature_sequence_extraction(self, input_tensor):
        """
        [私有] 特征序列提取
        对于图像的特征提取，通常使用卷积网络。这里借助了 VGG16 的卷积网络模型
        :param input_tensor: 输入图像，shape=[B，H，W, 3]
        :return: 返回卷积后的图像特征，shape=[B, H/32, W/4, 512]  => [B, 1, W/4, 512]
        """

        is_training = True if self.__phase == 'train' else False

        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            net = slim.repeat(input_tensor, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3')
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv4')
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4')
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv5')
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
            net = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv6')
        return net

    def __map_to_sequence(self, input_tensor):
        """
        [私有] 将featureMap 转换成 序列
        老实说这里自己不是太明白为什么使用 tf.squeeze ，如果仅仅是拿掉axis=1维，
        那可以用 tf.reshape() 替代么？
        :param input_tensor: 经过卷积后的图像特征，shape=[B, 1, W/4, 512]
        :return:
        """
        shape = input_tensor.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input_tensor, axis=1)

    def __sequence_label(self, input_tensor, input_sequence_length):
        """
        [私有] 序列标签
        借助于 双向LSTM（栈类型） 对序列化数据进行处理【解释一下，为什么需要循环网络？？】
        :param input_tensor: 序列化的featureMap, shape=[batch, width, 512]
        :param input_sequence_length: 经过双向LSTM + FC之后的序列结果，shape=[width, batch, num_classes]
        :return:
        """
        with tf.variable_scope('LSTM_Layers'):
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)

            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_num])

            # Doing the affine projection
            w = tf.Variable(tf.truncated_normal([hidden_num, self.__num_classes], stddev=0.01), name="w")
            logits = tf.matmul(rnn_reshaped, w)

            # 这样会报错，是因为 batch_size = None么？
            # logits = tf.reshape(logits, [batch_size, -1, self.__num_classes])
            input_shape = tf.shape(input_tensor)
            input_batch = input_shape[0]
            logits = tf.reshape(logits, [input_batch, -1, self.__num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return rnn_out, raw_pred

    def build_network(self, images, sequence_length=None):
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(images)
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(input_tensor=cnn_out)
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(input_tensor=sequence, input_sequence_length=sequence_length)
        return net_out

    def network_loss(self, net_out, labels, sequence_length, learning_rate, global_step):
        """
        [公开] 计算误差 -- 只参与训练，不参与预测

        :param net_out: 是network的输出， shape = [width, batch, num_classes]
                输入（训练）数据，是一个三维float型的数据结构[max_time_step , batch_size , num_classes]
        :param labels: 基于sparseTensor的label
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
        :param sequence_length: 当前批次中每张图片的 Width//4 == width (为了支持变长)
        :return:
        """

        # CTC的loss，实际上是p(l|x)，l是要探测的字符串，x就是Bi-LSTM输出的x序列
        # 其实就是各个可能的PI的似然概率的和，这个求最大的过程涉及到前向和后向算法了，可以参见CRNN的CTC原理部分
        # 对！对！对！损失函数就是p(l|x)，似然概率之和，使丫最大化。：https://blog.csdn.net/luodongri/article/details/77005948
        cost = tf.reduce_mean(tf.nn.ctc_loss(labels=labels,
                                             inputs=net_out,
                                             sequence_length=sequence_length,
                                             ignore_longer_outputs_than_inputs=True))

        logger.debug("CTC_Loss损失函数构建完毕")

        #  把cost放到里面去
        tf.summary.scalar(name='train.Cost', tensor=cost)

        # 定义梯度下降算法，只有执行optimizer后，global_step的值才会+1
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)\
                .minimize(loss=cost, global_step=global_step)  # <--- 这个loss是CTC的似然概率值

        tf.summary.scalar(name='train.Learning_Rate', tensor=learning_rate)
        return cost, optimizer

    def network_validate(self, net_out, labels, sequence_length):
        """
        [公开] 网络验证 -- 不参与训练，只参与验证
        这步是在干嘛？是说，你LSTM算出每个时间片的字符分布，然后我用它来做Inference，也就是前向计算
        得到一个最大可能的序列，比如"我爱北京天安门"，然后下一步算编辑距离，和标签对比
        :param net_out:
        :param labels:
        :param sequence_length:
        :return:
        """

        # 在这里beam_width=10
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                          sequence_length=sequence_length,
                                                          beam_width=10,
                                                          merge_repeated=False)
        logger.debug("CTC_Beam_Search(不参与训练)构建完毕")

        # 这里做编辑距离的均值，为了能够了解误差有多大
        sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
        sequence_dist = tensor_util.print_tensor(sequence_dist, "计算完编辑距离")

        # 这个只是看错的有多离谱，并没有当做损失函数，CTC loss才是核心
        tf.summary.scalar(name='validate.Seq_Dist', tensor=sequence_dist)
        return decoded, sequence_dist
