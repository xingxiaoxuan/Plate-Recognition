# 图片预处理
import cv2
import numpy
import matplotlib.pyplot as plt
# 全黑的灰度图
gray0 = numpy.zeros((50, 50), dtype=numpy.uint8)
# cv2.imshow('0', gray0)
# 全白的灰度图
gray0[:, :] = 255
gray255 = gray0[:, :]
# cv2.imshow('255', gray255)
# 将灰度图转换成彩色图
Img_rgb = cv2.cvtColor(gray255, cv2.COLOR_GRAY2RGB)
# 将RGB通道全部置成0
Img_rgb[:, :, 0:3] = 0
# cv2.imshow('(0,0,0)', Img_rgb)
# cv2.waitKey(0)
plt.imshow(Img_rgb)
plt.show()

img_path = 'F:/pycharm/Plate Recognition/result/test1/test_1.png'
image = cv2.imread(img_path)
plt.imshow(image)
plt.show()

print(image.shape)
h = image.shape[0]
w = image.shape[1]
for i in range(h):
    for j in range(w):
        Img_rgb[i+10, j+10, 0:3] = image[i, j, 0:3]

plt.imshow(Img_rgb)
plt.show()
