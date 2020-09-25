#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 11:08
# @Author  : dly
# @File    : data_augment.py
# @Desc    : 数据增强

import glob
import cv2

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
    augtrain_path = base_train_path + 'images_aug'
    augmask_path = base_train_path + 'labels_aug'

    train_img, masks = data_num(train_path, mask_path)
    for data in range(len(train_img)):
        image = cv2.imread(train_img[data])
        mask = cv2.imread(masks[data])

        # 水平翻转
        augmented_1 = HorizontalFlip(p=1)(image=image, mask=mask)
        aug_image_1 = augmented_1['image']
        aug_mask_1 = augmented_1['mask']
        cv2.imwrite(augtrain_path + "/aug_img{}_{}.jpg".format(data, 1), aug_image_1)
        cv2.imwrite(augmask_path + "/aug_mask{}_{}.png".format(data, 1), aug_mask_1)

        # 垂直翻转
        augmented_2 = VerticalFlip(p=1)(image=image, mask=mask)
        aug_image_2 = augmented_2['image']
        aug_mask_2 = augmented_2['mask']
        cv2.imwrite(augtrain_path + "/aug_img{}_{}.jpg".format(data, 2), aug_image_2)
        cv2.imwrite(augmask_path + "/aug_mask{}_{}.png".format(data, 2), aug_mask_2)

        # 水平 + 垂直 翻转
        augmented_3 = Flip(p=1)(image=image, mask=mask)
        aug_image_3 = augmented_3['image']
        aug_mask_3 = augmented_3['mask']
        cv2.imwrite(augtrain_path + "/aug_img{}_{}.jpg".format(data, 3), aug_image_3)
        cv2.imwrite(augmask_path + "/aug_mask{}_{}.png".format(data, 3), aug_mask_3)

        if data % 100 == 0:
            print(data)


if __name__ == "__main__":
    main()
