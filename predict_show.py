#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 9:45
# @Author  : dly
# @File    : predict_show.py
# @Desc    :

from train import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from nets.unet_ori import _unet

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def predict(image_file, index, model, output_path, n_class, weights_path=None):
    # 预测单张图片

    if weights_path is not None:
        model.load_weights(weights_path)
    img = cv2.imread(image_file).astype(np.float32)
    img = np.float32(img) / 255
    pr = model.predict(np.array([img]))[0]
    pr = pr.reshape((256, 256, n_class)).argmax(axis=2)
    seg_img = np.zeros((256, 256), dtype=np.uint16)
    for c in range(n_class):
        seg_img[pr[:, :] == c] = c
    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_img = np.zeros((256, 256, 3), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            tmp = matches[int(seg_img[i][j])] // 3.14
            save_img[i][j][0] = tmp
            save_img[i][j][1] = tmp
            save_img[i][j][2] = tmp
    # cv2.imwrite(os.path.join(output_path, index + ".png"), save_img)
    plt.imshow(save_img)
    plt.show()


def predict_and_deal(image_file, index, model, output_path, n_class, weights_path=None):
    # 预测单张图片

    if weights_path is not None:
        model.load_weights(weights_path)
    img = cv2.imread(image_file).astype(np.float32)
    img = np.float32(img) / 255
    pr = model.predict(np.array([img]))[0]
    pr = pr.reshape((256, 256, n_class)).argmax(axis=2)
    seg_img = np.zeros((256, 256), dtype=np.uint16)
    for c in range(n_class):
        seg_img[pr[:, :] == c] = c
    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_img = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            save_img[i][j] = matches[int(seg_img[i][j])]

    print(len(save_img[save_img == 100]))
    print(len(save_img[save_img == 200]))
    print(len(save_img[save_img == 300]))
    print(len(save_img[save_img == 400]))
    print(len(save_img[save_img == 500]))
    print(len(save_img[save_img == 600]))
    print(len(save_img[save_img == 700]))
    print(len(save_img[save_img == 800]))

    save_img_sec = np.zeros((256, 256), dtype=np.uint16)
    flag = False
    for num in matches:
        tmp = len(save_img[save_img == num]) / 65536
        print(tmp)
        if tmp > 0.89:
            save_img_sec[:, :] = num
            flag = True
            break

    print(len(save_img_sec[save_img_sec == 800]))
    # cv2.imwrite(os.path.join(output_path, index + ".png"), save_img)
    # plt.imshow(save_img)
    # plt.show()


if __name__ == "__main__":
    file = '176138.tif'

    base_test_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'
    weights_path = './models/ep020-loss0.317-val_loss0.494_0.8242.h5'
    input_path = base_test_path + 'image/' + file
    output_path = base_test_path + 'labels/'
    n_class = 8

    index, _ = os.path.splitext(file)
    model = _unet(n_class)
    model.load_weights(weights_path)  # 读取训练的权重
    predict(input_path, index, model, output_path, n_class)
