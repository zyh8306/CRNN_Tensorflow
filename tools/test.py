
import os
import time
import json

import tensorflow as tf
import numpy as np

from config import config
from crnn_model import model
from utils import text_util
from utils import image_util
from utils import tensor_util

os.environ["CUDA_VISIBLE_DEVICES"]=""

_IMAGE_HEIGHT = 32

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string('charset_file', './charset6k.txt', 'Path to the charset txt file..')

tf.app.flags.DEFINE_string('test_dir', './data/', 'Path to the test.txt. ')

tf.app.flags.DEFINE_string('model_dir', './model/', 'Base directory for the model.')


FLAGS = tf.app.flags.FLAGS


# 获取测试集文件名和label
def get_filename_with_label(test_dir: str):
    label_path = os.path.join(test_dir, "test.txt")
    files_list = []
    label_list = []

    with open(label_path, "r", encoding='UTF_8') as file:
        for line in file:
            # 分割
            filename, _, label = line[:-1].partition(' ')

            files_list.append(filename.strip())
            label_list.append(label.strip())

    return files_list, label_list


def _inference_crnn_ctc():

    charset_file = FLAGS.charset_file
    test_dir = FLAGS.test_dir
    model_dir = FLAGS.model_dir

    # 获取字符库
    characters = text_util.get_charset(charset_file)

    # 定义输入张量
    input_image = tf.placeholder(dtype=tf.float32, shape=[1, _IMAGE_HEIGHT, None, 3])
    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    # initialise the net model
    net = model.CrnnNet(
        phase='Test',
        hidden_num=config.cfg.ARCH.HIDDEN_UNITS, # 256
        layers_num=config.cfg.ARCH.HIDDEN_LAYERS,# 2层
        num_classes=len(characters) + 1)

    with tf.variable_scope('shadow'):
        net_out = net.build_network(images=input_image)

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_length,
                                                             beam_width=10, merge_repeated=True)

    # 读取文件 -- image_list, text_list
    image_list, text_list = get_filename_with_label(test_dir)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(model_dir)

    # 统计
    num_true = 0

    with tf.Session(config=sess_config) as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        for idx, image_name in enumerate(image_list):
            # 读取图像
            image = image_util.read_image_file(image_name)
            input_image_list = image_util.resize_batch_image([image], 'RESIZE_FORCE', config.cfg.ARCH.INPUT_SIZE)

            # 获取宽高
            image = input_image_list[0]
            height = image.shape[0]
            width = image.shape[1]
            seq_len = np.array([width // 4], dtype=np.int32)

            preds = sess.run(ctc_decoded, feed_dict={input_image:input_image_list, input_sequence_length:seq_len})

            pred_text = tensor_util.sparse_tensor_to_str(preds[0], characters)[0]
            true_text = text_list[idx]

            if pred_text == true_text:
                num_true += 1

            print(" =================" + image_name + "================= ")
            print('预测值 {:s} vs 真实值 {:s}'.format(pred_text, true_text))

    prob = num_true/len(image_list)
    print("预测准确率为：%f" % prob)

if __name__ == '__main__':
    _inference_crnn_ctc()
