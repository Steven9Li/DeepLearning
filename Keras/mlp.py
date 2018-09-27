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
from keras.models import load_model
from keras import backend as K

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
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_class,activation='softmax'))
    #model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])
    #json_string = model.to_json()
    #print(json_string)
    history = model.fit(train_data, train_labels, epochs = 4, verbose = 2, validation_data=(test_data, test_labels))
    #model.save('mlp_model_1')

    #old_model = load_model('mlp_model_1')
    #get_3rd_layer_output = K.function([model.layers[0].input,K.learning_phase()],
    #                              [model.layers[2].output])
    #layer_output = get_3rd_layer_output([train_data,1])[0]
    #print(len(layer_output[0]))
    #history = model.fit(train_data, train_labels, epochs = 1, verbose = 1, validation_data=(test_data, test_labels))
    '''
    history = model.fit(train_data, train_labels, epochs = epochs, verbose = 1, validation_data=(test_data, test_labels))
    score = model.evaluate(test_data,test_labels,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''


