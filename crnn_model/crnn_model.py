#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-21 下午6:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : crnn_model.py
# @IDE: PyCharm Community Edition
"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.contrib import rnn
from local_utils import log_utils
from crnn_model import cnn_basenet
from local_utils.log_utils import  _p_shape

logger = log_utils.init_logger()

class ShadowNet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    def __init__(self, phase: str, hidden_nums: int, layers_nums: int, num_classes: int):
        """

        :param phase: 'Train' or 'Test'
        :param hidden_nums: Number of hidden units in each LSTM cell (block)
        :param layers_nums: Number of LSTM cells (blocks)
        :param num_classes: Number of classes (different symbols) to detect
        """
        super(ShadowNet, self).__init__()
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__num_classes = num_classes
        return

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value: str):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
        return

    # 输出3件套：一个Conv卷积 + Relu + 池化[2x2]
    # out_dims:输出的维度，其实就是卷积核的数量，我讨厌这个变量命名
    def __conv_stage(self, inputdata: tf.Tensor, out_dims: int, name: str=None) -> tf.Tensor:
        """ Standard VGG convolutional stage: 2d conv, relu, and maxpool

        :param inputdata: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the maxpooled output of the stage
        """ #out_channel=out_dims，就是隐藏层的个数啊，64这里是
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
        relu = self.relu(inputdata=conv)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool #这是CNN一个阶段定义：卷基层+Relu+池化

    def shape(self,tensor):
        return tensor.get_shape().as_list()

    #抽feature，用的cnn网络
    def __feature_sequence_extraction(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"
        # https://blog.csdn.net/Quincuntial/article/details/77679463
        '''
        在CRNN模型中，通过采用标准CNN模型（去除全连接层）中的卷积层和最大池化层来构造卷积层的组件。
        这样的组件用于从输入图像中提取序列特征表示。在进入网络之前，所有的图像需要缩放到相同的高度。
        然后从卷积层组件产生的特征图中提取特征向量序列，这些特征向量序列作为循环层的输入。
        具体地，特征序列的每一个特征向量在特征图上按列从左到右生成。这意味着第i个特征向量是所有特征图第i列的连接。
        在我们的设置中每列的宽度固定为单个像素。

        # 由于卷积层，最大池化层和元素激活函数在局部区域上执行，因此它们是平移不变的。
        因此，特征图的每列对应于原始图像的一个矩形区域（称为感受野），并且这些矩形区域与特征图上从左到右的相应列具有相同的顺序。
        如图2所示，特征序列中的每个向量关联一个感受野，并且可以被认为是该区域的图像描述符。
        '''
        :param inputdata: eg. batch*32*100*3 NHWC format
        :return:

          |
        Conv1  -->  H*W*64          #卷积后，得到的维度
        Relu1
        Pool1       H/2 * W/2 * 64  #池化后得到的维度
          |
        Conv2       H/2 * W/2 * 128
        Relu2
        Pool2       H/4 * W/4 * 128
          |
        Conv3       H/4 * W/4 * 256
        Relu3
          |
        Conv4       H/4 * W/4 * 256
        Relu4
        Pool4       H/8 * W/4 * 64
          |
        Conv5       H/8 * W/4 * 512
        Relu5
        BatchNormal5
          |
        Conv6       H/8 * W/4 * 512
        Relu6
        BatchNormal6
        Pool6       H/16 * W/4 * 512
          |
        Conv7
        Relu7       H/16 * W/4 * 512
          |

          20层

        """
        logger.debug("CNN层的输入inputdata的Shape:%r",self.shape(inputdata))

        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64

        logger.debug("CNN层第1层输出的Shape:%r", self.shape(conv1))

        conv2 = self.__conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128

        logger.debug("CNN层第2层输出的Shape:%r", self.shape(conv2))

        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3')  # batch*8*25*256
        relu3 = self.relu(conv3) # batch*8*25*256

        logger.debug("CNN层第3层输出的Shape:%r", self.shape(relu3))

        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*25*256
        relu4 = self.relu(conv4)  # batch*8*25*256
        # 这里诡异啊，池化用的[2,1]，一般都是正方形池化啊
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*25*256

        logger.debug("CNN层第4层输出的Shape:%r", self.shape(max_pool4))

        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*25*512
        relu5 = self.relu(conv5)  # batch*4*25*512
        if self.phase.lower() == 'train':
            bn5 = self.layerbn(inputdata=relu5, is_training=True)
        else:
            bn5 = self.layerbn(inputdata=relu5, is_training=False)  # batch*4*25*512

        logger.debug("CNN层第5层输出的Shape:%r", self.shape(bn5))

        conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6')  # batch*4*25*512
        relu6 = self.relu(conv6)  # batch*4*25*512
        if self.phase.lower() == 'train':
            bn6 = self.layerbn(inputdata=relu6, is_training=True)
        else:
            bn6 = self.layerbn(inputdata=relu6, is_training=False)  # batch*4*25*512
        max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512

        logger.debug("CNN层第6层输出的Shape:%r", self.shape(max_pool6))

        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7')  # batch*1*25*512
        #？？？怎么就从batch*2*25*512=>batch*1*25*512了？只是个卷基层啊？晕了
        relu7 = self.relu(conv7)  # batch*1*25*512

        logger.debug("CNN层第7层输出的Shape:%r", self.shape(relu7))

        return relu7

    def __map_to_sequence(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Implements the map to sequence part of the network.

        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the lenght of the sequences that the LSTM expects
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        logger.debug("inputdata的shape: %r",shape)
        assert shape[1] == 1  # H of the feature map must equal to 1
        # 数据本身是 [N,H,W,512],删掉后，就变成了[N,W,512]了么？？？
        # 不对啊，这块，H结果是1维度，才可以squeeze啊，这不可能啊，H不可能维度是1啊？！
        # tf.squeeze()：把是1的维度压缩掉，如果不指定axis，压缩掉所有是1的维度；如果指定axis，压缩掉指定的axis。
        return self.squeeze(inputdata=inputdata, axis=1)
        # 我觉得这块有问题！！！，应该是把H高度，当做批次的一部分，训练完再reshape回来，这块得去再看看别人的代码？？？


    # 这个是往LSTM里面灌
    # 输入：
    #       我理解是 [N , W , 512]
    # 输出：
    #       应该是概率矩阵把
    def __sequence_label(self, inputdata: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Implements the sequence label part of the network

        :param inputdata:
        :return:
        """
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # forward lstm cell
            # __hidden_nums = 256，__layers_nums=2
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                         dtype=tf.float32)
            # Bi-LSTM，输入是[N*H , W , 512]，输出是[N*H , W , 512]
            # 为何呢？因为隐含层是256，输出是256维度，但是由于是Bi-LSTM，俩LSTM要concat到一起，得~，变成512了又

            if self.phase.lower() == 'train':#dropout好像只能在某个方向上丢来着？？？忘了，这个LSTM的正则化还得去回忆下
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]

            # [ N*H*W, 512 ]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

            # 在做一个全连接，隐含层个数是类别数，类别是啥，就是字典里面字的数量
            w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
            # Doing the affine projection

            logits = tf.matmul(rnn_reshaped, w) # 全连接

            logits = tf.reshape(logits, [batch_s, -1, self.__num_classes]) #reshape回来，折腾啥呢

            #看！做softmax呢，小样！我一直等着你呢，卧槽，还求了个argmax，得到最大的可能的那个字符了？
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis 转置(1, 0, 2)，=>列没变，行和高置换，也就是 （batch,rnn length,class)=>（rnn length,batch,class)
            # transpose(0,1,2)=>(1,0,2)，就是维度0和维度1互换了，呢，变成了[W, N*H, Cls]
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred #返回的是一个张量（rnn length,batch,class)，和每个rnn步骤 预测的最可能的字符

    def build_shadownet(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Main routine to construct the network

        :param inputdata:
        :return:
        """
        # first apply the cnn feature extraction stage
        # 是一个20层的卷积网络，返回的是 [W/16,H/4,512] 的结果
        # 对了，输入的高度被归一化成32了
        cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

        cnn_out = _p_shape(cnn_out,"CNN网络抽取了特征后的输出")


        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)



        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(inputdata=sequence)

        net_out = _p_shape(net_out, "LTSM的输出net_out")
        raw_pred = _p_shape(raw_pred, "LTSM的输出raw_pred")

        return net_out
