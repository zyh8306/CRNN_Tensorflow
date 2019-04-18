from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import os,math
import cv2
from local_utils import data_utils

'''
2019.4.18 这个文件已经废弃，
请转到crnn_generator.py，which 使用ctpn的生成代码，达到复用目的，
'''

# #############
# 设置各种的参数：
# #############

DEBUG=True
ROOT="data_generator"   # 定义运行时候的数据目录，原因是imgen.sh在根部运行
DATA_DIR="data"
MAX_LENGTH=30   # 可能的最大长度（字符数），大概可以到512个像素，这个是我们设计的最宽的宽度
MIN_LENGTH=5    # 可能的最小长度（字符数）
MAX_HEIGHT=35   # 最大的高度（像素）
MIN_HEIGHT=26   # 最大的高度（像素）

# 颜色的算法是，产生一个基准，然后RGB上下浮动FONT_COLOR_NOISE
MAX_FONT_COLOR = 150    # 最大的可能颜色
FONT_COLOR_NOISE = 10   # 最大的可能颜色
ONE_CHARACTOR_WIDTH = 28# 一个字的宽度
ROTATE_ANGLE = 4        # 随机旋转角度
ROTATE_POSSIBLE = 0.4   # 按照定义的概率比率进行旋转，也就是100张里可能有多少个发生旋转
GAUSS_RADIUS_MIN = 0.8  # 高斯模糊的radius最小值
GAUSS_RADIUS_MAX = 1.3  # 高斯模糊的radius最大值
POSSIBILITY_ROTOATE = 0.4   # 文字的旋转概率，40%

# 仿射的倾斜的错位长度  |/_/, 这个是上边或者下边右移的长度
AFFINE_OFFSET = 12


# 从文字库中随机选择n个字符
def generate_words():
    start = random.randint(0, len(info_str)-MAX_LENGTH-1)
    length = random.randint(MIN_LENGTH,MAX_LENGTH)
    end = start + length
    random_word = info_str[start:end]
    if DEBUG: print("截取内容[%s]，%d" %(random_word,length))
    return random_word,length

# 产生随机颜色
def random_word_color():
    base_color = random.randint(0, MAX_FONT_COLOR)
    noise_r = random.randint(0, FONT_COLOR_NOISE)
    noise_g = random.randint(0, FONT_COLOR_NOISE)
    noise_b = random.randint(0, FONT_COLOR_NOISE)

    noise = np.array([noise_r,noise_g,noise_b])
    font_color = (np.array(base_color) + noise).tolist()

    if DEBUG: print('font_color：',font_color)

    return tuple(font_color)

# 生成一张图片
def create_backgroud_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path+bground_choice)
    if DEBUG: print('background:',bground_choice)
    if DEBUG: print(bground.size[0],bground.size[1])
    if DEBUG: print("width, height: %d,%d" % (width, height))

    # 在大图上随机产生左上角
    x = random.randint(0,bground.size[0]-width)
    y = random.randint(0,bground.size[1]-height)
    bground = bground.crop((x, y, x+width, y+height))

    return bground


# 随机接受概率
def _random_accept(accept_possibility):
    return np.random.choice([True,False], p = [accept_possibility,1 - accept_possibility])


def _rotate_one_point(xy, center, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    cord = (# (xy[0] - center[0]) * cos_theta - (xy[1]-center[1]) * sin_theta + xy[0],
            # (xy[0] - center[0]) * sin_theta + (xy[1]-center[1]) * cos_theta + xy[1]
            (xy[0] - center[0]) * cos_theta - (xy[1] - center[1]) * sin_theta + center[0],
            (xy[0] - center[0]) * sin_theta + (xy[1] - center[1]) * cos_theta + center[1])
    # print("旋转后的坐标：")
    # print(cord)
    return cord


def _rotate_points(points,center, degree):
    theta = math.radians(-degree)
    original_min_x, original_min_y = np.array(points).max(axis=0)
    rotated_points = [_rotate_one_point(xy, center, theta) for xy in points]
    rotated_min_x, rotated_min_y = np.array(rotated_points).max(axis=0)
    x_offset = abs(rotated_min_x - original_min_x)
    y_offset = abs(rotated_min_y - original_min_y)
    rotated_points = [(xy[0]+x_offset, xy[1]+y_offset) for xy in rotated_points]
    return rotated_points


# 随机仿射一下，也就是歪倒一下
# 不能随便搞，我现在是让图按照平行方向歪一下，高度不变，高度啊，大小啊，靠别的控制，否则，太乱了
def random_affine(img):

    HEIGHT_PIX = 10
    WIDTH_PIX = 50

    # 太短的不考虑了做变换了
    # print(img.size)
    original_width = img.size[0]
    original_height = img.size[1]
    points = [(0,0), (original_width,0), (original_width,original_height), (0,original_height)]

    if original_width<WIDTH_PIX: return img,points
    # print("!!!!!!!!!!")
    if not _random_accept(POSSIBILITY_AFFINE): return img,points

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)

    is_top_fix = random.choice([True,False])

    bottom_offset = random.randint(0,AFFINE_OFFSET) # bottom_offset 是 上边或者下边 要位移的长度

    height = img.shape[0]

    # 这里，我们设置投影变换的3个点的原则是，使用    左上(0,0)     右上(WIDTH_PIX,0)    左下(0,HEIGHT_PIX)
    # 所以，他的投影变化，要和整个的四边形做根据三角形相似做换算
    # .
    # |\
    # | \
    # |__\  <------投影变化点,  做三角形相似计算，offset_ten_pixs / bottom_offset =  HEIGHT_PIX / height
    # |   \                   所以： offset_ten_pixs = (bottom_offset * HEIGHT_PIX) / height
    # |____\ <-----bottom_offset
    offset_ten_pixs = int(HEIGHT_PIX * bottom_offset / height)   # 对应10个像素的高度，应该调整的横向offset像素
    width = int(original_width  + bottom_offset )#

    pts1 = np.float32([[0, 0], [WIDTH_PIX, 0], [0, HEIGHT_PIX]])  # 这就写死了，当做映射的3个点：左上角，左下角，右上角


    #\---------\
    # \         \
    #  \_________\
    if is_top_fix:  # 上边固定，意味着下边往右
        # print("上边左移")
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(0,0),
                  (original_width,0),
                  (width,original_height),
                  (bottom_offset,original_height)]
    #  /---------/
    # /         /
    #/_________/
    else:  # 下边固定，意味着上边往右
        # 得先把图往右错位，然后
        # 先右移
        # print("上边右移")
        H = np.float32([[1, 0, bottom_offset], [0, 1, 0]])  #
        img = cv2.warpAffine(img, H, (width, height))
        # 然后固定上部，移动左下角
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [-offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(bottom_offset,0),
                  (original_width+bottom_offset,0),
                  (width,original_height),
                  (0,original_height)]


    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    return img,points

# 旋转函数
def random_rotate(img,points):
    ''' ______________
        |  /        /|
        | /        / |
        |/________/__|
        旋转可能有两种情况，一种是矩形，一种是平行四边形，
        但是传入的points，都是4个顶点，
    '''
    if not _random_accept(POSSIBILITY_ROTOATE): return img,points # 不旋转

    w,h = img.size

    center = (w//2,h//2)

    if DEBUG: print("需要旋转")
    degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
    if DEBUG: print("旋转度数:%f" % degree)
    return img.rotate(degree,center=center,expand=1),_rotate_points(points,center,degree)


def add_noise(img):
    # img = (scipy.misc.imread(filename)).astype(float)
    noise_mask = np.random.poisson(img)
    noisy_img = img + noise_mask
    return noisy_img

# 模糊函数
def darken_func(image):
    #.SMOOTH
    #.SMOOTH_MORE
    #.GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数
    radius = random.uniform(GAUSS_RADIUS_MIN,GAUSS_RADIUS_MAX)
    filter_ = random.choice(
                            [ImageFilter.SMOOTH,
                             ImageFilter.DETAIL,
                            # ImageFilter.SMOOTH_MORE, # 效果太过分，也删掉
                            ImageFilter.GaussianBlur(radius=radius),
                            ImageFilter.EDGE_ENHANCE,      #边缘增强滤波 
                            # ImageFilter.EDGE_ENHANCE_MORE, 这个效果太过分了，删除掉
                            ImageFilter.SHARPEN] #为深度边缘增强滤波
                            )
    if DEBUG: print("模糊函数：%s" % str(filter_))
    image = image.filter(filter_)
    return image


def random_resize(image,width,height):
    if DEBUG: print("调整前[%d,%d]" %(width,height))

    height = int(random.randint(MIN_HEIGHT,MAX_HEIGHT)) # 随机调整图像高度
    width = int(width * (height/MAX_HEIGHT)) # 宽度也对应调整
    image = image.resize((width,height))

    if DEBUG: print("调整后[%d,%d]" % (width, height))
    return image

# 老版本，废弃
# # 旋转函数
# def random_rotate(image):
#
#     #按照5%的概率旋转
#     rotate = np.random.choice([True,False], p = [ROTATE_POSSIBLE,1 - ROTATE_POSSIBLE])
#     if not rotate:
#         return image
#     if DEBUG: print("需要旋转")
#     degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
#     image = image.rotate(degree)
#
#     return image


# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size,len):
    width, height = bground_size
    #if DEBUG: print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size*len)
    y = random.randint(0, int((height-font_size)/4))

    return x, y

# 随机生成文字
def generate_words(charset):
    length = random.randint(MIN_LENGTH,MAX_LENGTH)
    s = ""
    for i in range(length):
        j = random.randint(0, len(charset))
        s += charset[j]
    if DEBUG: print("随机生成的汉字字符串[%s]，%d" %(s,length))
    return s,length

def random_font_size():
    font_size = random.randint(24,27)

    return font_size

def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font

def main(save_path, num, label_file):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    charset = data_utils.get_charset("charset6k.txt")

    # 随机选取10个字符
    random_word,len = generate_words(charset)

    width = ONE_CHARACTOR_WIDTH * len
    height = MAX_HEIGHT

    # 生成一张背景图片，已经剪裁好，宽高为32*280
    # raw_image = create_backgroud_image(ROOT+'/background/', 280, 32),28一个字
    raw_image = create_backgroud_image(ROOT + '/background/', width, height)

    # 随机选取字体大小
    font_size = random_font_size()
    # 随机选取字体
    font_name = random_font(ROOT+'/font/')
    # 随机选取字体颜色
    font_color = random_word_color()

    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = random_x_y(raw_image.size, font_size,len)

    # 将文本贴到背景图片
    if DEBUG: print(font_name)
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)

    # 随机选取作用函数和数量作用于图片
    raw_image = darken_func(raw_image)
    raw_image = random_rotate(raw_image)
    raw_image = random_resize(raw_image,width, height)

    # 保存文本信息和对应图片名称
    image_file_name = str(num) + '.png'
    save_path = os.path.join(save_path, image_file_name)
    if DEBUG: print("文件名：%s" % save_path)

    label_file.write(save_path + " " + random_word + '\n')

    raw_image.save(save_path)


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--dir")
    parser.add_argument("--num")

    args = parser.parse_args()

    DATA_DIR = args.dir
    TYPE= args.type

    # 处理具有工商信息语义信息的语料库，去除空格等不必要符号
    # with open(ROOT+'/info.txt', 'r', encoding='utf-8') as file:
    #     info_list = [part.replace('\t', '') for part in file.readlines()] # \t不能显示正常，删掉
    #     info_str = ''.join(info_list)

    # 同时生成label，记录下你生成标签文件名
    label_file_name = os.path.join(DATA_DIR,TYPE+".txt")
    label_file = open(label_file_name, 'w', encoding='utf-8')
    total = int(args.num)

    # 生成图片数据
    label_dir = os.path.join(DATA_DIR,TYPE)
    for num in range(0,total):
        main(label_dir, num, label_file)
        if DEBUG: print("--------------------")
        if num % 1000 == 0:
            print('生成了样本[%d/%d]'%(num,total))


