# import d2lzh as d2l
import math
import numpy as np
# from mxnet import nd

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    print("s1:%r" % s1)
    x1 -= eta / math.sqrt(s1 + eps) * g1
    print("x1:%r" % x1)
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
# d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

x1 = np.array([1.0,1.0,1.0])
x2 = np.array([2.0,2.0,2.0])
s1 = np.array([3.0,3.0,3.0])
s2 = np.array([4.0,4.0,4.0])
adagrad_2d(x1,x2,s1,s2)