# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:09:41 2019

@author: gear
github:  https://github.com/gear106
"""
import numpy as np
import pandas as pd
from model import TSRCNN

#------------------------load_data-------------------------------#

def main():
    '这里训练数据集形式如下'
    train_X = np.random.rand(100,16,16,1) # 这里表示100个16x16的二维数据
    train_Y = np.random.rand(100,64,64,1) # 假设上采样4x
    tsrcnn = TSRCNN(lr_size=16, saved_result=saved_model, is_training=True, learning_rate=1e-4, batch_size=16, epochs=1)
    tsrcnn.train(train_X, train_Y)
    
if __name__ == '__main__':
    
    saved_model = 'TVSR.h5'  # 保存模型名称
    main()
    
    
    
    
