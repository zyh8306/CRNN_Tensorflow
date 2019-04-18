import unittest

from local_utils import data_utils

class TestDict(unittest.TestCase):

    def test_init(self):
        pass

    def test_get_charset(self):
        charset = data_utils.get_charset("charset.txt")
        self.assertEquals(len(charset),5991)

        charset = data_utils.get_charset("charset6k.txt")
        print(len(charset))
        self.assertEquals(len(charset),6883)

    def find_not_words(self):
        charset = open("charset.txt", 'r', encoding='utf-8').readlines()
        charset = [ch.strip("\n") for ch in charset]
        for c in charset:
            if len(c)==1:
                if ord(c)<0x4E00 or ord(c)>= 0x9FA5:
                    print(c)
            else:
                print("错误：%s" % c)
if __name__ == '__main__':
    unittest.main()
    # t = TestDict()
    # t.find_not_words()
