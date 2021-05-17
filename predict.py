from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1',
            'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin', 'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu',
            'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong', 'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang',
            'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun', 'zh_zang', 'zh_zhe']

image = np.zeros((20, 20))
# image_path = 'F:/pycharm/plate-recognition-master/146c69bd/车牌检测识别数据集/训练字符识别模型数据/8/2-0.jpg'
image_path = "F:/pycharm/Plate Recognition/result/test1/test_1.png"
img = Image.open(image_path).convert('L')

for h in range(0, 20):
    for w in range(0, 20):
        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
        if img.getpixel((w, h)) > 230:
            image[w][h] = 0
        else:
            image[w][h] = 1

image = image.reshape((1, 20, 20, 1))
# img = image.load_img(image_path, target_size=(20, 20))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.

model = load_model('cnn1.h5')

predict = model.predict(image)
print(predict)
i = predict.argmax()
print(template[i])

plt.imshow(img)
plt.show()
