#!/usr/bin/env python
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
imgName = "1.png";
image = Image.open(imgName)
filter_ = ImageFilter.GaussianBlur(radius=0.5)
image = image.filter(filter_)
image.save("out/0.5.out.png")
filter_ = ImageFilter.GaussianBlur(radius=0.6)
image = image.filter(filter_)
image.save("out/0.6.out.png")
filter_ = ImageFilter.GaussianBlur(radius=0.7)
image = image.filter(filter_)
image.save("out/0.7.out.png")
filter_ = ImageFilter.GaussianBlur(radius=0.8)
image = image.filter(filter_)
image.save("out/0.8.out.png")
filter_ = ImageFilter.GaussianBlur(radius=0.9)
image = image.filter(filter_)
image.save("out/0.9.out.png")
filter_ = ImageFilter.GaussianBlur(radius=1)
image = image.filter(filter_)
image.save("out/0.1.out.png")
filter_ = ImageFilter.GaussianBlur(radius=1.1)
image = image.filter(filter_)
image.save("out/0.1.1.out.png")

image = image.filter(ImageFilter.SMOOTH)
image.save("out/0.smooth.out.png")

image = image.filter(ImageFilter.DETAIL)
image.save("out/0.detail.out.png")

W=512
H=32
import cv2


# 给图片加白色padding，先resize
def padding(image):

    h,w,c = image.shape
    print("original image:%d,%d" % (h,w))
    x_scale = W/w
    y_scale = H/h
    if x_scale<y_scale:
        y_scale = x_scale
    else:
        x_scale = y_scale

    print("x,y sale:%f,%f" % (x_scale,y_scale))
    # https://www.jianshu.com/p/11879a49d1a0 关于resize
    image =  cv2.resize(image, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)

    h, w, c = image.shape

    # top,bottom,left,right对应边界的像素数目
    top = bottom = int((H - h) /2)
    left = right = int((W - w) /2)

    image = cv2.copyMakeBorder(image, top,bottom,left,right, cv2.BORDER_REPLICATE)

    print("resized image:",image.shape)
    return image



if __name__=="__main__":
    image = cv2.imread("0.png")
    image = padding(image)
    cv2.imwrite("./out/0.padding.png",image)