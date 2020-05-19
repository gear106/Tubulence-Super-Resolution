# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:37:38 2019

@author: gear
github:  https://github.com/gear106
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    data = x_data.reshape(size, size, size)    
    train_x = []

    for i in range(0, size-box_size+1, dist):
        for j in range(0, size-box_size+1, dist):
            for k in range(0, size-box_size+1, dist):
                tmp = data[i:i+box_size, j:j+box_size, k:k+box_size]
                tmp = tmp.reshape(box_size,box_size,box_size,1)
                train_x.append(tmp)
    
    train_x = np.array(train_x)
                
    return train_x

def load_patch_data(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    temp = x_data.reshape(size, size, size)    
    
    # 处理边界问题
    s = mesh_size + dist
    m = np.int16(dist/2)
    data = np.zeros((s, s, s))
    print(data.shape)
    data[m:-m, m:-m, m:-m] = temp
    train_x = []

    for i in range(0, s-box_size+1, dist):
        for j in range(0, s-box_size+1, dist):
            for k in range(0, s-box_size+1, dist):
                tmp = data[i:i+box_size, j:j+box_size, k:k+box_size]
                tmp = tmp.reshape(box_size,box_size,box_size,1)
                train_x.append(tmp)
    
    train_x = np.array(train_x)
                
    return train_x
                
def data_patch(root, train_or_test='train', mesh_size=64, box_size=16, dist=16):
    
    u_name = os.listdir(root+'u')
    v_name = os.listdir(root+'v')
    w_name = os.listdir(root+'w')
    print(u_name)
    numbers = len(u_names)
    
    x = []
    for i in range(numbers):
         
        load_Fu =  pd.read_table(root+'u/'+u_name[i], header=None, sep='\s+').values    
        load_Fv =  pd.read_table(root+'v/'+v_name[i], header=None, sep='\s+').values    
        load_Fw =  pd.read_table(root+'w/'+w_name[i], header=None, sep='\s+').values
    
        test_Fu = load_data(load_Fu, mesh_size, box_size, dist)
        test_Fv = load_data(load_Fv, mesh_size, box_size, dist)
        test_Fw = load_data(load_Fw, mesh_size, box_size, dist)
        
        if train_or_test == 'train':
            
            x.append(test_Fu)
            x.append(test_Fv)
            x.append(test_Fw)
            
        elif train_or_test == 'test':
            test_x = np.concatenate([test_Fu, test_Fv, test_Fw], axis=-1)
            x.append(test_x)
            
    test_X = np.concatenate(x, axis=0)
 
    return test_X


def train_dataset(root):
    train_X = data_patch(root+'FDNS/', mesh_size=64, box_size=16, dist=16)
    train_Y = data_patch(root+'DNS/', mesh_size=512, box_size=128, dist=128)
    
    return train_X, train_Y


def test_dataset(root):
    test_X = data_patch(root+'FDNS/', mesh_size=64, box_size=16, dist=16)
    test_Y = data_patch(root+'DNS/', mesh_size=512, box_size=128, dist=128)
    
    return test_X, test_Y

def les_dataset(root):
    test_X = test_data_patch(root+'FDNS/', mesh_size=64, box_size=16, dist=8)
    
    return test_X

    
root = 'E:/TSRCNN/data_0305/512/gauss_r4/'
train_x, train_y = test_dataset(root)
test_x, test,y = train_dataset(root)

root_save = 'E:/TSRCNN/dataset/'
np.save(root_save+'train_x/train_x_512_gauss_r4.npy',train_x)
np.save(root_save+'train_y/train_y_512_gauss_r4.npy',train_y)

np.save(root_save+'test_x/test_x_512_gauss_r4_t3.npy',train_x)
np.save(root_save+'test_y/test_y_256_gauss_r4_t3.npy',train_y)
            
                        
            
            
            
    
