from train import *
import cv2
import numpy as np

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
    save_img = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            save_img[i][j] = matches[int(seg_img[i][j])]
    cv2.imwrite(os.path.join(output_path, index + ".png"), save_img)


def predict_all(input_path, output_path, model, n_class, weights_path=None):
    # 预测一个文件夹内的所有图片
    # input_path：传入图像文件夹
    # output_path：保存预测图片的文件夹
    # model：传入模型
    # n_class：类别数量
    # weights_path：权重保存路径
    if weights_path is not None:
        model.load_weights(weights_path)
    for image in os.listdir(input_path):
        print(image)
        index, _ = os.path.splitext(image)
        predict(os.path.join(input_path, image),
                index, model, output_path, n_class)


if __name__ == "__main__":
    base_test_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/test/'
    weights_path = './models/ep005-loss0.528-val_loss0.559_val_acc0.79.h5'
    input_path = base_test_path + 'image_A/'
    output_path = base_test_path + 'labels/'
    n_class = 8

    model = mobilenet_unet(n_class)
    model.load_weights(weights_path)  # 读取训练的权重
    predict_all(input_path, output_path, model, n_class, weights_path)
