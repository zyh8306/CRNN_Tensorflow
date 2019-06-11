import logging
import os
import numpy as np
from utils import text_util
from utils import image_util

logger = logging.getLogger("data producer")


class DataProducer:

    @staticmethod
    def work(dir_path, data_type, charsets):
        def read_label_lines(in_file_path):
            in_file = open(label_file_path, "r", encoding='UTF-8')
            text_lines = []
            try:
                text_lines = in_file.readlines()
            finally:
                in_file.close()
            return text_lines

        # 这里应该是读取txt
        label_file_path = os.path.join(dir_path, data_type) + ".txt"

        # 获取labels行数
        label_lines = read_label_lines(label_file_path)
        logger.info("find %d images in %s", len(label_lines), label_file_path)
        label_index_list = np.arange(0, len(label_lines))

        while True:
            np.random.shuffle(label_index_list)

            try:
                for idx in label_index_list:

                    # 从文件中读取样本路径和标签值
                    # data/train/21.png 你好北京
                    line = label_lines[idx]
                    filename, _, label = line[:-1].partition(' ')

                    # 读取图像
                    # if filename.find('data\\') != -1:
                    #     filename = filename.replace('data\\', '')
                    # image_path = os.path.join(dir_path, filename)
                    image_path = filename
                    image = image_util.read_image_file(image_path)
                    # 读取label
                    label = text_util.convert_label_to_id(label, charsets)

                    if label is None:
                        continue

                    # logger.info("%s 文件读取成功", filename)
                    yield image, label

            except Exception as ex:
                logger.exception("DataProducor Exception:", ex)
                continue


