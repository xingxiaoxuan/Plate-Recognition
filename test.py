# 图片预处理
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = 'F:/pycharm/Plate Recognition/result/test1/test_6.png'


def img_resize_to_target_black(image):
    # 全黑的灰度图
    gray0 = np.zeros((100, 100), dtype=np.uint8)
    # cv2.imshow('0', gray0)
    # 全白的灰度图
    gray0[:, :] = 255
    gray255 = gray0[:, :]
    # cv2.imshow('255', gray255)
    # 将灰度图转换成彩色图
    Img_rgb = cv2.cvtColor(gray255, cv2.COLOR_GRAY2RGB)
    # 将RGB通道全部置成0
    Img_rgb[:, :, 0:3] = 255
    h = image.shape[0]
    w = image.shape[1]
    for i in range(h):
        for j in range(w):
            Img_rgb[i + 30, j + 40, 0:3] = 255 - image[i, j, 0:3]

    return Img_rgb


image = cv2.imread(img_path)
img = img_resize_to_target_black(image)

plt.imshow(img)
plt.show()

cv2.imwrite('F:/pycharm/Plate Recognition/result/test1/test10.png', img)
