# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:46:17 2020

@author: gear
github:  https://github.com/gear106
"""
import pandas as pd
import numpy as np
from model import TSRCNN
import matplotlib.pyplot as plt
from concatenate import concat
from scipy.io import FortranFile



def main(test_u, test_v, test_w, test_y):  
    tsrcnn = TSRCNN(lr_size=p_size, channels=1, saver=savers, is_training=False)
    tmp_u = []
    tmp_v = []
    tmp_w = []
    for i in range(nums):     
        print((tsrcnn.process(test_u[i]).shape))
        tmp_u.append(tsrcnn.process(test_u[i]))
        tmp_v.append(tsrcnn.process(test_v[i]))
        tmp_w.append(tsrcnn.process(test_w[i]))
    pred_u = concat(tmp_u, sub=p_size, size=nums)
    pred_v = concat(tmp_v, sub=p_size, size=nums)
    pred_w = concat(tmp_w, sub=p_size, size=nums)
    plt.figure(figsize=(7,7))
    plt.imshow(pred_u[0], cmap=plt.cm.RdBu_r)          # 显示预测的 u 方向速度切面
    plt.figure(figsize=(7,7))
    plt.imshow(test_y[0,0,:,:,0], cmap=plt.cm.RdBu_r)  # 显示真实的 u 方向速度切面
    
    np.savetxt('./results/fft_pred_u_256_cut_p8_0414.txt', pred_u.reshape(-1))
    np.savetxt('./results/fft_pred_v_256_cut_p8_0414.txt', pred_v.reshape(-1))
    np.savetxt('./results/fft)pred_w_256_cut_p8_0414.txt', pred_w.reshape(-1))

              
if __name__ == '__main__':
    
    m_size = 128     # FDNS数据网格大小
    p_size = 16      # FNDS 切块大小
    nums = int(m_size / p_size)   # 切块个数 
    
    savers = './weight/tvsr_weight_gauss_r4_0416.h5'    # 模型保存路径
    root = 'E:/TSRCNN/dataset/'
    test_x = np.load(root+'test_x/test_x_512_gauss_t3.npy') 
    test_y = np.load(root+'test_y/test_y_512_gauss_t3.npy') 
    
    test_u = test_x[:,:,:,:,0].reshape(nums,1,p_size,p_size,p_size,1)
    test_v = test_x[:,:,:,:,1].reshape(nums,1,p_size,p_size,p_size,1)
    test_w = test_x[:,:,:,:,2].reshape(nums,1,p_size,p_size,p_size,1)
    
    root_x = 'E:/TSRCNN/data_0305/512/cut/FDNS/u/'
    load_Fu =  pd.read_table(root_x +'Flow_FDNS_u_3D10001.dat', header=None, sep='\s+').values
    test_fu = load_Fu.reshape(m_size,m_size,m_size)
    plt.figure(figsize=(7,7))
    plt.imshow(test_fu[0], cmap=plt.cm.RdBu_r)   #显示FNDS速度场
    plt.show()
    main(test_u, test_v, test_w, test_y)