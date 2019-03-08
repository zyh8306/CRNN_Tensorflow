from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import os

'''
1. 从文字库随机选择10个字符
2. 生成图片
3. 随机使用函数
'''

# #############
# 设置各种的参数：
# #############

DEBUG=True
ROOT="data_generator"   # 定义运行时候的数据目录，原因是imgen.sh在根部运行
DATA_DIR="data"
MAX_LENGTH=12   # 可能的最大长度（字符数）
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

# 从文字库中随机选择n个字符
def sto_choice_from_info_str():
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
    x, y = random.randint(0,bground.size[0]-width), random.randint(0, bground.size[1]-height)
    bground = bground.crop((x, y, x+width, y+height))

    return bground


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


# 旋转函数
def random_rotate(image):

    #按照5%的概率旋转
    rotate = np.random.choice([True,False], p = [ROTATE_POSSIBLE,1 - ROTATE_POSSIBLE])
    if not rotate:
        return image
    if DEBUG: print("需要旋转")
    degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
    image = image.rotate(degree)

    return image

# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size,len):
    width, height = bground_size
    #if DEBUG: print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size*len)
    y = random.randint(0, int((height-font_size)/4))

    return x, y

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

    # 随机选取10个字符
    random_word,len = sto_choice_from_info_str()

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
    with open(ROOT+'/info.txt', 'r', encoding='utf-8') as file:
        info_list = [part.replace('\t', '') for part in file.readlines()] # \t不能显示正常，删掉
        info_str = ''.join(info_list)

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
            if DEBUG: print('[%d/%d]'%(num,total))
    file.close()


