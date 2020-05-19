# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:09:41 2019

@author: gear
github:  https://github.com/gear106
"""
import pandas as pd
import numpy as np
from model import TVSR



def main(savers):
    
    train_X = np.load(train_x)
    train_Y = np.load(train_y)
    
    tsrcnn = TVSR(lr_size=8, channels=1, saver=savers, is_training=True, learning_rate=1e-4, 
                    batch_size=16, epochs=200)
    history = tsrcnn.train(train_X, train_Y)
    df_results = pd.DataFrame(history.history)
    df_results['epoch'] = history.epoch
    df_results.to_csv(path_or_buf='./history/History_tvsr_gauss_r4_0416.csv',index=False)
    
if __name__ == '__main__':
    
    p_size = 16   # 切块大小
    savers = './weight/tvsr_weight_gauss_r4_0416.h5'   # 模型保存地址
    #------------------------load_data-------------------------------#
    filter_type = 'gauss_r4'
    root = 'E:/TSRCNN/dataset/'
    train_x = root+'train_x/%s/train_x_%s.npy'%(filter_type,filter_type)
    train_y = root+'train_y/%s/train_y_%s.npy'%(filter_type,filter_type)
    main(savers)
    
    
    
    
