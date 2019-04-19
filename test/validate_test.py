import unittest

from local_utils import data_utils
import tools.validate as val


class TestValidate(unittest.TestCase):

    def test_init(self):
        pass

    def test_get_charset(self):
        # 每个可以匹配上2个字

        # 都是5个长，好算数用
        preds = [
            "我不爱空空",
            "门天空空空",
            "12我是来",
            "fizz好",
        ]
        #都是10个长，好算数
        labels = [
            "我爱北京天安门占占占",
            "天安门上太阳升占占占",
            "1234567890",
            "abcdefghij",
        ]

        #  2x4行/ 10x4行 = 0.2
        self.assertEquals(0.2,val.caculate_precision(preds,labels))

if __name__ == '__main__':
    unittest.main()
