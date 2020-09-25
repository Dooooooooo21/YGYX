from nets.unet import mobilenet_unet
from nets.unet_ori import _unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import numpy as np
import cv2 as cv

NCLASSES = 8
HEIGHT = 256
WIDTH = 256
base_train_path = 'C:/Users/Dooooooooo21/Desktop/project/YGYX/train/'

# 类别对应
matches = [100, 200, 300, 400, 500, 600, 700, 800]


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(',')[0]
            # 从文件中读取图像
            img = cv.imread(base_train_path + 'image_aug/' + name)
            img = img.astype(np.int32)
            # img = img.resize((WIDTH, HEIGHT))
            img = img / 127.5 - 1
            X_train.append(img)

            name = (lines[i].split(',')[1]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open(base_train_path + 'label_aug/' + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            seg_labels = np.zeros((WIDTH, HEIGHT, NCLASSES))

            for m in matches:
                img[img == m] = matches.index(m)

            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img == c).astype(int)

            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))


if __name__ == "__main__":
    model_dir = "models/"
    # 获取model
    model = _unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    model.load_weights('./models/ep020-loss0.317-val_loss0.494_0.8242.h5')

    model.summary()
    # 打开数据集的txt
    with open(r"./data/train_data.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(2333)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 95%用于训练，5%用于估计
    num_val = int(len(lines) * 0.05)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
        model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=1
    )
    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    batch_size = 12
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr, early_stopping])

    model.save_weights(model_dir + 'last1.h5')
