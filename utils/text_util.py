import re
import numpy as np

rex = re.compile(' ')


# 加载字符集，charset.txt，最后一个是空格
def get_charset(charset_file):
    charset = open(charset_file, 'r', encoding='utf-8').readlines()
    charset = [ch.strip("\n") for ch in charset]
    charset = "".join(charset)
    charset = list(charset)
    return charset


# 处理一些“宽”字符
def process_unknown_charactors(sentence, dict):
    unkowns = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／"
    knows = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/"

    result = ""
    for one in sentence:
        # 对一些特殊字符进行替换，替换成词表的词
        i = unkowns.find(one)
        if i==-1:
            letter = one
        else:
            letter = knows[i]
            # logger.debug("字符[%s]被替换成[%s]", one, letter)

        # 看是否在里面
        if letter not in dict:
            # logger.error("句子[%s]的字[%s]不属于词表,剔除此样本",sentence,letter)
            return None

        result+= letter
    return result


# 将label转换为数字表示
def convert_label_to_id(label, charsets):
    # 获取label内容
    # 1.label预处理校验
    label = process_unknown_charactors(label, charsets)
    # 2.非空校验
    if label is None:
        return None
    # 3.去除空格
    label = rex.sub('', label)
    # 4.将label转为数字
    label = [charsets.index(l) for l in label]
    return label


# 按照List中最大长度扩展label
def extend_to_max_len(labels, ext_val: int = -1):
    max_len = 0
    for one in labels:
        if len(one)>max_len:
            max_len = len(one)

    for one in labels:
        one.extend( [ext_val] * (max_len - len(one)) )

    return np.array(labels, dtype=np.int32)


if __name__ == "__main__":
    charset = get_charset("../charset6k.txt")

    label_id = convert_label_to_id('我爱北京天安门', charset)
    for id in label_id:
        print(id, end=",")
    print("\n")

    print("=======将label数组扩展===========")
    label_id2 = convert_label_to_id('天津', charset)
    labels = []
    labels.append(label_id)
    labels.append(label_id2)
    res = extend_to_max_len(labels)
    for item in labels:
        for id in item:
            print(id, end=",")
        print("")

