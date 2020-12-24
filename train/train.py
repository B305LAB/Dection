# coding: utf-8
from __future__ import print_function
import os
import numpy as np
import cv2

import warnings
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras import backend as K
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard


train_path1 = "../raw_data/round_train/part1/OK_Images/"
test_path1 = "../raw_data/round_test/part1/TC_Images/"  # 产品1的测试集路径
model_path1 = '../model/model1/'
temp_path = '../temp_data/'
save_path1 = '../temp_data/data/focusight1_round2_train_part1/TC_Images/'
result_path = '../result/'


def load_data(train_path):
    trainfiles = os.listdir(train_path)
    trainfile_names = []
    train = np.empty([len(trainfiles), 128, 128])
    for i in range(len(trainfiles)):
        trainfile_names.append(os.path.splitext(trainfiles[i])[0])
        img = cv2.imread(train_path + trainfiles[i], cv2.IMREAD_GRAYSCALE)
        train[i] = img
    return train, trainfile_names




'''构建模型'''
autoencoder = Sequential([
    Conv2D(filters=30, kernel_size=[3, 3], strides=[2, 2], padding='same', input_shape=[128, 128, 1],
           activation='relu'),
    Conv2D(filters=60, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 64*64*30
    Conv2D(filters=90, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 32*32*60
    Conv2D(filters=120, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 16*16*90
    Conv2DTranspose(90, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 32x32x120
    Conv2DTranspose(60, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  ## # 32x32x60
    Conv2DTranspose(30, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # # 32x32x30
    Conv2DTranspose(1, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # # 32x32x1
])
autoencoder.summary()
autoencoder.compile(loss=keras.losses.mean_squared_error,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
''''''''''''''''''''''''''''''''

def random_point():
    return np.random.randint(0, 128, size=[2, ])


def random_numer(low, high):
    if low == high:
        high += 1
    return np.random.randint(min(low, high), max(low, high))
def make(img, area=300, min_p=166, max_p=222):
    res = np.copy(img)
    LP = random_point()
    while (res[LP[0], LP[1]] < 20):
        LP = random_point()
    H = random_numer(5, 10)
    W = area // H
    RP = LP + [H, W]
    RP[0] = min(127, RP[0])
    RP[1] = min(127, RP[1])
    flag = np.zeros_like(res)
    for i in range(100):
        Y = random_numer(LP[0], RP[0])
        X = random_numer(LP[1], RP[1])
        res[Y, X] = random_numer(min_p, max_p)
        flag[Y, X] = 1
    res2 = np.copy(res)
    for y in range(LP[0], RP[0]):
        for x in range(LP[1], RP[1]):
            if y + 3 > 127 or x + 3 > 127:
                continue
            tile = flag[y:y + 3, x:x + 3]
            filter = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ])
            if np.sum(tile * filter) > 0:
                res2[y, x] = random_numer(min_p, max_p)
    return res2


def load_XY(X):
    X = np.concatenate([X, X, X], axis=0)
    x_train = None
    y_train = None
    diff = [(70, 75), (80, 85), (200, 225)]
    for i in range(len(X)):
        img = X[i]
        if np.random.rand() % 1 > 0.5:
            rand_int = random_numer(0, 3)
            img_quexian = make(img, *diff[rand_int])
        else:
            img_quexian = np.copy(img)
        if x_train is None:
            x_train = img_quexian[np.newaxis, :, :, np.newaxis]
        else:
            x_train = np.concatenate([x_train, img_quexian[np.newaxis, :, :, np.newaxis]], axis=0)
        if y_train is None:
            y_train = img[np.newaxis, :, :, np.newaxis]
        else:
            y_train = np.concatenate([y_train, img[np.newaxis, :, :, np.newaxis]])
    index = np.array([i for i in range(0, len(X))])
    np.random.shuffle(index)
    return X[:,:,:,np.newaxis], y_train[index]

def train(X, model, model_path):
    X = X.astype('float32')
    x_t, y_t = load_XY(X)
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.2)
    model.fit(x_train, y_train,
              epochs=200,
              batch_size=32,
              shuffle=True,
              validation_data=(x_test, y_test))
    model.save_weights(model_path + 'model.h5')

train1, trainfile_names1 = load_data(train_path1)

train(train1, autoencoder, model_path1)
