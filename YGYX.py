#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 15:37
# @Author  : dly
# @File    : YGYX.py
# @Desc:

import matplotlib.image as mpig
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

base_train_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'
base_test_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/test/'


# 查看数据
def img_data():
    # plt
    # content = plt.imread(basepath + 'label/5.png')

    # matplotlib.image
    # content = mpig.imread(basepath + 'label/5.png')

    # PIL
    content = np.array(Image.open(base_train_path + 'label/5.png'))

    # keras.preprocessing.image
    # content = load_img(basepath + 'label/5.png')
    # content = img_to_array(content)
    print(content)


# 数据生成器
def data_generator(train_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_dir, batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir, batch_size=32, class_mode='categorical')

    return train_generator, test_generator


data_generator(base_train_path + 'image', base_test_path)
