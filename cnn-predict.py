import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

path = 'F:/pycharm/Plate Recognition/image/test2.png'
origin_image = cv2.imread(path)


# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destoryAllWindows()


# cv_show("origin picture", origin_image)

# plt 显示彩色图片
def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt_show0(origin_image)


# 显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# plt_show(origin_image)  # 有问题，显示的应该是热力图像


# 图像去噪灰度图像
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# gray_image = gray_guss(origin_image)
# plt_show(gray_image)


# 提取车牌部分图片
def get_carLicense_img(image):
    gray_image = gray_guss(image)
    # 利用Sobel算子进行图像梯度计算 src：输入图像 ddepth:输出图像深度
    # dx,dy:1,0 求x方向一阶导数， 0,1 求y方向一阶导数
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    # 转回 uint8
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # print(ret) # 大津法求阈值进行二值化
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    # 进行图像闭运算 将白色部分连在一起
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=3)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    # 中值滤波去除噪点
    image = cv2.medianBlur(image, 15)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 3)) and (weight < (height * 4)):
            image = origin_image[y:y + height, x:x + weight-5]
            return image


image = origin_image.copy()
carLicense_image = get_carLicense_img(image)
plt_show0(carLicense_image)
plt.show()


# 车牌字符分割
def carLicense_spilte(image):
    gray_image = gray_guss(image)

    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    plt_show(image)
    # 计算二值图像黑白点的个数，处理绿牌照问题，让车牌号码始终为白色
    area_white = 0
    area_black = 0
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    if area_white > area_black:
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        plt_show(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    # 排序 按照x从小到大排序
    words = sorted(words, key=lambda s: s[0], reverse=False)
    print(words)
    i = 0
    for word in words:
        if (word[3] > (word[2] * 1.8)) and (word[3] < (word[2] * 3.5)):
            i = i + 1
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(splite_image)
    return word_images


image = carLicense_image.copy()
word_images = carLicense_spilte(image)

# 绿牌要改为8，蓝牌为7，显示所用
for i, j in enumerate(word_images):
    plt.subplot(1, 8, i + 1)
    plt.imshow(word_images[i], cmap='gray')
    # cv2.imwrite('F:/pycharm/Plate Recognition/result/test1/test_' + str(i) + '.png', word_images[i])
plt.show()

template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1',
            'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin', 'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu',
            'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong', 'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang',
            'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun', 'zh_zang', 'zh_zhe']

results = []
for i in range(8):
    image = word_images[i]
    image = cv2.resize(image, (20, 20))
    image = image.reshape((1, 20, 20, 1))

    model = load_model('cnn1.h5')
    predict = model.predict(image)
    r = predict.argmax()
    results.append(r)
    result = template[r]
    print(result)

print(results)
