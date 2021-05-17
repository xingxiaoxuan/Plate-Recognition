# cnn识别数字、字母、省份65个字符
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import xlsxwriter
from keras import models, layers, optimizers
from keras.utils import to_categorical

# 选择CPU0，可以注释掉
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

# 数据集地址
SAVER_DIR = r"F:/pycharm/plate-recognition-master/146c69bd/车牌检测识别数据集/训练字符识别模型数据"
NUM_CLASSES = 65  # 分为65种可能性
SIZE = 400  # 输入图片大小为20*20
WIDTH = 20  # 图片长20、宽20
HEIGHT = 20

list_path = os.listdir(SAVER_DIR)  # 根目录下的文件路径组成列表
# print(list_path)
# input_num = np.zeros(shape=(65,))
train_count = 0

# 计算输入图片总个数
for i in range(NUM_CLASSES):
    train_list = SAVER_DIR + "/" + list_path[i]
    # print(train_list)
    for rt, dirs, files in os.walk(train_list):
        for filename in files:
            train_count += 1
            # input_num[i] = input_count
            # print("%s的字符数据集个数" %list_path[i], input_count)

# input_images = np.array([[0] * train_count for i in range(SIZE)])
input_images = np.zeros((49063, 20, 20))
input_labels = np.array([[0] * train_count for i in range(NUM_CLASSES)])
train_labels = np.zeros((1, train_count))
print("input_images.shape: ", input_images.shape)  # (400, 49063) 输入49063张图片，400个像素点
print("input_labels.shape: ", input_labels.shape)  # (65, 49063) 输入49063张图片，65种标签

index = 0
for i in range(NUM_CLASSES):
    train_list = SAVER_DIR + "/" + list_path[i]
    # print(train_list)
    for rt, dirs, files in os.walk(train_list):
        for filename in files:
            # 读取子文件夹下所有图片
            filename = os.path.join(train_list, filename)
            img = Image.open(filename).convert('L')
            width = img.size[0]
            height = img.size[1]
            # 制作图片数据集
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        # input_images[w + h * width][index] = 0
                        input_images[index][w][h] = 0
                    else:
                        # input_images[w + h * width][index] = 1
                        input_images[index][w][h] = 1
            input_labels[i][index] = 1
            train_labels[0][index] = i  # 制作标签
            index += 1

print("数据集训练结束，开始训练网络")
print("input_images: ", input_images.shape)  # (49063, 20, 20)
print("train_labels: ", train_labels.shape)  # (65, 49063)

# f = xlwt.Workbook()
# sheet1 = f.add_sheet('input_images')
# for j in range(400):
#     for i in range(49063):
#         sheet1.write(i, j, int(input_images[i, j]))
# f.save(r'F:/pycharm/Plate Recognition/image/input.xls')

# 保存数据，但是数据集太大保存不全，超出范围
# workbook = xlsxwriter.Workbook('input.xlsx')  # 新建excel表
# worksheet1 = workbook.add_worksheet('train_image')  # 新建sheet（sheet的名称为"train_image"）
# worksheet2 = workbook.add_worksheet('train_label')
# for i in range(49063):
#     worksheet2.write(0, i, int(train_labels[0, i]))
#     for j in range(400):
#         worksheet1.write(j, i, int(input_images[j, i]))
# workbook.close()  # 将excel文件保存关闭，如果没有这一行运行代码会报错

train_images = input_images.reshape((49063, 20, 20, 1))
# train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
# 去除多余维度
train_labels = np.squeeze(train_labels)

# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# test_labels = to_categorical(test_labels)

# 训练模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(65, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=20, batch_size=64)
# model.save('cnn1.h5')  # 保存模型

print(history)

# 绘图 损失和精准度
acc = history.history['acc']
loss = history.history['loss']
epoch = range(1, len(acc) + 1)
plt.plot(epoch, acc, 'b', label='acc')
plt.plot(epoch, loss, 'r', label='loss')
plt.legend()
plt.show()
