#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 11:08
# @Author  : dly
# @File    : data_augment.py
# @Desc    : 数据增强

import glob
import cv2
from PIL import Image
import numpy as np

from albumentations import (
    Flip,
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    RandomRotate90
)

base_train_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'


# 获取要进行数据增强的训练集和标签
def data_num(train_path, mask_path):
    train_img = glob.glob(train_path)
    masks = glob.glob(mask_path)
    return train_img, masks


# compuse 组合变换
def mask_aug():
    aug = Compose([
        HorizontalFlip(p=1),
        Flip(p=1)
    ])
    return aug


def main():
    # 输入
    train_path = base_train_path + 'image/*.tif'
    mask_path = base_train_path + 'label/*.png'

    # 增强结果输出目录
    augtrain_path = base_train_path + 'image_aug/'
    augmask_path = base_train_path + 'label_aug/'

    train_img, masks = data_num(train_path, mask_path)
    for data in range(len(train_img)):
        file_name = train_img[data].split('\\')[1].split('.')[0]
        image = cv2.imread(train_img[data])
        mask = np.array(Image.open(masks[data]))

        # 水平翻转
        augmented_1 = HorizontalFlip(p=1)(image=image, mask=mask)
        aug_image_1 = augmented_1['image']
        aug_mask_1 = Image.fromarray(augmented_1['mask'])
        cv2.imwrite(augtrain_path + "/{}_{}.tif".format(file_name, 1), aug_image_1)
        aug_mask_1.save(augmask_path + "/{}_{}.png".format(file_name, 1))

        # 垂直翻转
        augmented_2 = VerticalFlip(p=1)(image=image, mask=mask)
        aug_image_2 = augmented_2['image']
        aug_mask_2 = Image.fromarray(augmented_2['mask'])
        cv2.imwrite(augtrain_path + "/{}_{}.tif".format(file_name, 2), aug_image_2)
        aug_mask_2.save(augmask_path + "/{}_{}.png".format(file_name, 2))

        # 水平 + 垂直 翻转
        augmented_3 = Transpose(p=1)(image=image, mask=mask)
        aug_image_3 = augmented_3['image']
        aug_mask_3 = Image.fromarray(augmented_3['mask'])
        cv2.imwrite(augtrain_path + "/{}_{}.tif".format(file_name, 3), aug_image_3)
        aug_mask_3.save(augmask_path + "/{}_{}.png".format(file_name, 3))

        if data % 1000 == 0:
            print(data)


if __name__ == "__main__":
    main()
