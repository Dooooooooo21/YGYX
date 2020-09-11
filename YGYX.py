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
from tensorflow.keras.preprocessing.image import load_img, img_to_array

basepath = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'


def img_data():
    # plt
    # content = plt.imread(basepath + 'label/5.png')

    # matplotlib.image
    # content = mpig.imread(basepath + 'label/5.png')

    # PIL
    content = np.array(Image.open(basepath + 'label/5.png'))

    # keras.preprocessing.image
    # content = load_img(basepath + 'label/5.png')
    # content = img_to_array(content)
    print(content)


img_data()
