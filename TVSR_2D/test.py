# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:36:45 2019

@author: gear
github:  https://github.com/gear106
"""
import numpy as np
from model import TSRCNN
import matplotlib.pyplot as plt
from load_data import test_dataset



def main():
    test_X = np.random.rand(1,16,16,1) # 这里表示1个16x16的二维测试数据
    test_Y = np.random.rand(1,64,64,1) # 假设上采样4x
    model_test = TSRCNN(lr_size=16, saved_result=test, is_training=False)
    pred_Y = model_test.process(test_X)
    plt.imshow(pred_Y.reshape(64,64)) # 显示重构结果
    plt.show()
    plt.imshow(test_Y.reshape(64,64)) # 显示对应标签
                 
if __name__ == '__main__':
    test = 'TVSR.h5'
    main()
