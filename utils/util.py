#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 16:25
# @Author  : dly
# @File    : util.py
# @Desc    : 文件拷贝


import os
import shutil


def file_copy(src, dst):
    file_list = os.listdir(src)

    for file in file_list:
        shutil.copy(src + file, dst + file)


base_train_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'
file_copy(base_train_path + 'image/', base_train_path + 'image_aug/')
file_copy(base_train_path + 'label/', base_train_path + 'label_aug/')
