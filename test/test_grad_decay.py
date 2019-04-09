import numpy as np
import matplotlib.pyplot as plt

PRINT_INTERVAL = 500

# 测试线性衰减，参考花书 181页
def linear_decay(grad,epslon_0 = 0.1,epslon_T = 0.00001):
    print("\n线性衰减算法：epslon=%f" % epslon_0)
    print("---------------------------------")
    print("%20s%20s" % ("delta", "g"))

    T = len(grad)
    for k, g in enumerate(grad):
        alpha = k / T
        epslon_k = (1-alpha) * epslon_0  +  alpha * epslon_T
        delta = epslon_k * g
        if k % PRINT_INTERVAL == 0:
            print("%20f,g=%20f" % (delta, g))


# 花书 p180
def sgd(grad,epslon=0.01):
    print ("\nSGD算法：epslon=%f" % epslon)
    print("---------------------------------")
    print("%20s%20s" % ("delta", "g"))

    # 我设置这个序列都是1
    for k,g in enumerate(grad):
        delta = - epslon * g
        if k % PRINT_INTERVAL == 0:
            print("%20f%20f" % (delta,g))


def momentum(grad,epslon = 0.001,alpha=0.9):
    print("\nMomentum算法：epslon=%f" % epslon)
    print("---------------------------------")
    print("%20s%20s" % ("delta", "g"))

    v = 0

    # 我设置这个序列都是1
    for k,g in enumerate(grad):
        v = alpha * v - epslon * g # 算法是-，但是，那个是负梯度方向，所以这里改成+
        if k % PRINT_INTERVAL == 0:
            print("%20f%20f" % (v, g))

def AdaGrad(grad,epslon=0.01,sigma=0.0000001):
    print("\nAdaGrad算法：epslon=%f" % epslon)
    print("---------------------------------")
    print("%20s%20s%20s" % ("delta", "g","decay"))

    r=0
    for k,g in enumerate(grad):
        r = r + g*g
        decay = 1/(sigma + np.sqrt(r))
        delta = - epslon * decay * g
        if k % PRINT_INTERVAL == 0:
            print("%20f%20f%20f" % (delta, g,decay))

# https://zh.d2l.ai/chapter_optimization/adadelta.html
# http://xudongyang.coding.me/gradient-descent-variants/
def AdaDelta(grad,epslon=0.01,rou=0.9,sigma=0.0000001):
    print("\nAdaDelta算法：epslon=NAN，不依赖于学习率")
    print("---------------------------------")
    print("%20s%20s%20s" % ("delta", "g","decay"))

    sigma = 0.000000001
    s = 0
    delta_x = 0
    delta_x_t_1 = 0

    for k,g in enumerate(grad):

        s = rou * s + (1-rou) * g * g

        _g =  np.sqrt( (delta_x_t_1 + sigma) / (s + sigma)) * g

        delta_x = rou * delta_x_t_1 + (1-rou) * _g * _g

        decay = np.sqrt(delta_x_t_1 + sigma)/ np.sqrt(s + sigma)

        delta = g * decay

        delta_x_t_1 = delta_x

        if k % PRINT_INTERVAL == 0:
            print("%20f%20f%20f" % (delta, g , decay))



def RMSPro(grad,epslon=0.01,rou=0.9,sigma=0.0000001):
    print("\nRMSPro算法：epslon=%f" % epslon)
    print("---------------------------------")
    print("%20s%20s%20s" % ("delta", "g","decay"))

    r=0

    for k,g in enumerate(grad):
        r = r * rou + g * g * (1-rou)
        decay = 1 / (np.sqrt(sigma + r))
        delta = - epslon * decay * g
        if k % PRINT_INTERVAL == 0:
            print("%20f%20f%20f" % (delta, g, decay))

def Adam(grad,epslon=0.01,rou1=0.9, rou2=0.99, sigma=0.0000001):
    print("\nAdam算法：epslon=%f" % epslon)
    print("---------------------------------")
    print("%20s%20s%20s" % ("delta", "g","decay"))

    s,r = 0,0
    for k,g in enumerate(grad):
        s = s * rou1 + g * (1-rou1)
        r = r * rou2 + g * g * (1 - rou2)
        decay = 1 / (np.sqrt(sigma + r))
        delta = - epslon * s * decay
        if k % PRINT_INTERVAL == 0:
            print("%20f%20f%20f" % (delta, g, decay))


def generate_grad(step):

    # 为了便于观察，生成1上下波动的梯度
    mu=1
    sigma = 0.1

    #生成梯度的序列，正态的
    #grad = np.random.normal(loc=mu, scale=sigma, size=step)

    # 尝试生成一个模拟真实梯度递减的一个序列
    grad = np.exp(np.linspace(-10,5,step))[::-1]
    print(grad)
    print("产生一个指数衰减的梯度：数量[%d],最大梯度[%20f]，最小梯度[%20f]" % (len(grad), grad[0],grad[-1]))

    return grad


def draw(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.show()

STEP = 10000
EPSLON = 0.1


# 产生梯度序列，模拟一个梯度序列，只有一个维度，便于理解
grad = generate_grad(10000)

#draw(linear_decay())

sgd(grad,0.1)
# sgd(grad,0.5)
# sgd(grad,5)
# linear_decay(grad,EPSLON)
# momentum(grad,0.1,0.9)
# AdaGrad(grad,EPSLON)
RMSPro(grad,0.1)
Adam(grad,0.1)
# Adam(grad,0.01)
# Adam(grad,0.001)
# Adam(grad,0.000001)
AdaDelta(grad)