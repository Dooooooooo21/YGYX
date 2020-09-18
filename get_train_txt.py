#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 10:57
# @Author  : dly
# @File    : get_train_txt.py
# @Desc:

import os

base_train_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'

tifs = os.listdir(base_train_path + 'image/')
pngs = os.listdir(base_train_path + 'label/')

with open('./data/train_data.txt', 'w') as f:
    count = 0
    for tif in tifs:
        # if count >= 50000:
        #     break
        png = tif.replace('tif', 'png')
        if png in pngs:
            f.write(tif + ',' + png + '\n')
            count += 1
