import tensorflow as tf
import numpy as np

inputdata = tf.placeholder(dtype=tf.float32,
                           shape=[64, None, 5990],
                           name='input')
decodes, prob = tf.nn.ctc_beam_search_decoder(inputs=inputdata,
                                              beam_width = 10,
                                              sequence_length= np.array(10 * [64]),
                                              merge_repeated=False)
# sequence_length: 1-D `int32` vector containing sequence lengths,having size `[batch_size]`.
# 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。