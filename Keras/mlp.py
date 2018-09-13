'''
使用keras搭建最简单的mlp网络
掌握keras的基本用法
'''
import numpy as np
from Mnist import *

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

if(__name__=='__main__'):

    # 数据文件路径
    train_data_path = 'D:\Data\MNIST\\train-images.idx3-ubyte'
    train_label_path = 'D:\Data\MNIST\\train-labels.idx1-ubyte'
    test_data_path = 'D:\Data\MNIST\\t10k-images.idx3-ubyte'
    test_label_path = 'D:\Data\MNIST\\t10k-labels.idx1-ubyte'

    #基本参数
    batch_size = 128
    num_class = 10
    epochs = 20

    #获取并归一化数据
    train_data = analysis_data(train_data_path).astype('float32')/255
    test_data = analysis_data(test_data_path).astype('float32')/255
    print(test_data.shape)
    train_labels = keras.utils.to_categorical(analysis_data(train_label_path))
    test_labels = keras.utils.to_categorical(analysis_data(test_label_path))
    print(test_labels.shape)

    #建立模型
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_class,activation='softmax'))
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])
    history = model.fit(train_data, train_labels, epochs = epochs, verbose = 1, validation_data=(test_data, test_labels))
    score = model.evaluate(test_data,test_labels,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


