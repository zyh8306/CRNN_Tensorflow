import logging
import time

import sys

from local_utils import data_utils
from utils.data_factory import DataFactory

logger = logging.getLogger("data test")

if __name__ == '__main__':

    charset6k = data_utils.get_charset('../charset6k.txt')

    gen = DataFactory.get_batch(data_dir='D:\workspace\gitCDC\ocr\hang\CRNN_Tensorflow\data', charsets=charset6k)
    while True:
        images= next(gen)

        # 图像 处理
        # A -- 直接Resize( H, W)
        # B -- 获取批次里面最大值，然后Resize
        # C -- 直接使用图片本身的大小

        image_width = [ image[0].shape[1] for image in images]
        time.sleep(1)

        print('images size = ' + str(len(images)))
