'''
mnist数据集解析文件
将mnist数据集解析成相应的numpy矩阵

2051数据文件
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 

2049标签文件
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 

'''

import numpy as np
import struct
import os

def path_exist(path):
    if os.path.exists(path):
        return 1
    else:
        print("The path %s does not exist!" % path)
        return 0

def analysis_data(path):
    f = open(path,'rb')
    data = f.read(4)
    digit = int.from_bytes(data, byteorder='big', signed=True)
    images = []
    #print(digit)
    if digit == 2051:
        num_of_img = int.from_bytes(f.read(4), byteorder='big', signed=True)
        num_of_rows = int.from_bytes(f.read(4), byteorder='big', signed=True)
        num_of_cols = int.from_bytes(f.read(4), byteorder='big', signed=True)
        res = np.frombuffer(f.read(num_of_img * num_of_rows * num_of_cols), dtype = np.uint8)
        return res.reshape(num_of_img, num_of_cols*num_of_rows)
    labels = []
    if digit == 2049:
        num_of_img = int.from_bytes(f.read(4), byteorder='big', signed=True)
        res = np.frombuffer(f.read(num_of_img), dtype = np.uint8)
        return res

    print("illegal file")

        

if(__name__=='__main__'):
    path = 'D:\Data\MNIST'
    path_exist(path)
    file_list = os.listdir(path)
    #train_data_path = 'D:\Data\MNIST\\train-images.idx3-ubyte'
    #train_label_path = 'D:\Data\MNIST\\train-labels.idx1-ubyte'
    test_data_path = 'D:\Data\MNIST\\t10k-images.idx3-ubyte'
    #test_label_path = 'D:\Data\MNIST\\t10k-labels.idx1-ubyte'
    s = analysis_data(test_data_path)