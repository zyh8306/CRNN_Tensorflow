import numpy as np

'''
    这是一个沙盒程序，专门用来测试各类小函数
'''
import cv2


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = [] # 位置,哪些位置上非0
    values = [] # 具体的值

    for n, seq in enumerate(sequences): # sequences是一个二维list
        indices.extend(zip([n]*len(seq), range(len(seq)))) # 生成所有值的坐标，不管是不是0，都存下来
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences),np.asarray(indices).max(0)[1]+1],
                       dtype=np.int64) # shape的行就是seqs的个数，列就是最长的那个seq的长度

    return indices, values, shape


def test_sparse_tuple_from():
    data = [
        [1,2,3],
        [4,5],
        [6,7,8,9,10,11,12]
    ]
    a,b,c = sparse_tuple_from(data)
    print(a)
    print(b)
    print(c)

def expand_array(data):
    max = 0
    for one in data:
        if len(one)>max:
            max = len(one)

    for one in data:
        one.extend( [0] * (max - len(one)) )

    return data

def test_expand_array():
    data = [
        [1,2,3],
        [4,5],
        [6,7,8,9,10,11,12]
    ]
    new_data = expand_array(data)
    print (new_data)


def process_unknown_charactors(sentence):
    unkowns = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／"
    knows = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/"

    result = ""
    for one in sentence:

        i = unkowns.find(one)

        if i == -1:
            letter = one
        else:
            letter = knows[i]

        result += letter
    return result


if __name__ == '__main__':
    # test_sparse_tuple_from()
    #test_expand_array()
    print(process_unknown_charactors("０我爱北京［天安门］Ａ"))
