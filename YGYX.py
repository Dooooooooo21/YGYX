#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 15:37
# @Author  : dly
# @File    : YGYX.py
# @Desc:

import cv2 as cv
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
    content = np.array(Image.open(base_train_path + 'label_aug/101_2.png'))

    # content = cv.imread(base_train_path + 'images/1_3_2.tif')

    # keras.preprocessing.image
    # content = load_img(base_train_path + 'label/5.png')
    # content = img_to_array(content)
    print(content)


img_data()

