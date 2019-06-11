import multiprocessing
import time
import queue
import threading
import logging
import numpy as np
import sys

from utils.data_producer import DataProducer

logger = logging.getLogger("data factory")

"""
数据工厂，所有的数据都是（由工人制造），再由工厂对外销售的。
"""


class DataFactory:

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        """
        :param generator:
        :param use_multiprocessing:
        :param wait_time:
        :param random_seed:
        :return:
        """
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self._queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):

        # 生产数据任务
        def data_generator_task(name):
            try:
                while not self._stop_event.is_set():
                    if self._use_multiprocessing or self._queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        # logger.info("%s produce an image. ", name)
                        self._queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
            except Exception:
                self._stop_event.set()
                raise

        try:
            if self._use_multiprocessing:
                self._queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self._queue = queue.Queue()
                self._stop_event = threading.Event()

            for i in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task, args=("进程_" + str(i),))
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task, args=("线程_" + str(i),))

                self._threads.append(thread)
                thread.start()
        except Exception as ex:
            logger.exception("DataFactory start Exception:", ex)
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def is_empty(self):
        return self._queue.empty()

    def stop(self, timeout=None):
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self._queue is not None:
                self._queue.close()

        self._threads = []
        self._stop_event = None
        self._queue = None

    def get(self):
        while self.is_running():
            if not self._queue.empty():
                inputs = self._queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)

    @staticmethod
    def get_batch(data_dir, charsets, data_type='train', batch_size=32, num_workers=4, use_multiprocessing=True):
        """ 获取数据
        :param data_dir: 数据集的路径
        :param charsets: 字符集
        :param data_type: 选择数据集的类型，默认是train
        :param batch_size: 默认32
        :param num_workers: 进程(或线程)并发数
        :param use_multiprocessing: 是否选择用进程
        :return: 返回一个generator

        需要注意的是：在Windows中，只能用 num_workers=1, use_multiprocessing=False
        参考：https://github.com/matterport/Mask_RCNN/issues/13
        """

        # 为了用着方便，修正一下
        if sys.platform.startswith('win32'):
            num_workers = 1
            use_multiprocessing = False
            logger.info("当前系统是Windows系统，修正参数num_workers=1，use_multiprocessing=False")

        # 定义一个工厂
        factory = DataFactory(DataProducer.work(data_dir, data_type, charsets), use_multiprocessing)

        try:
            # 工厂开始工作
            factory.start(max_queue_size=24, workers=num_workers)

            # 开始从工厂里面获取数据
            while True:
                image_list = []
                label_list = []
                image_size = 0
                while factory.is_running():
                    if not factory.is_empty():
                        # 读取数据
                        image, label = next(factory.get())
                        image_list.append(image)
                        label_list.append(label)

                        image_size += 1
                        # logger.debug("从 DataFactory 中获取数据")

                        if image_size>=batch_size:
                            break
                    else:
                        time.sleep(0.01)

                # yield一调用，就挂起，等着外面再来调用next()了
                yield [image_list, label_list]

                image_list = []
                label_list = []
                image_size = 0
        except Exception as ex:
            factory.stop()
            raise
        finally:
            if factory is not None:
                factory.stop()
