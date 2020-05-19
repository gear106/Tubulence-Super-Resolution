# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:39:37 2020

@author: gear
github:  https://github.com/gear106
"""
import numpy as np

#将预测的数据拼接（最大只能预测128**3的数据，内存限制（64G)所以需要拼接）
def concat(data, sub=64, size=2):
    '''
    sub: 切块尺寸
    size: 沿某一坐标轴切块个数
    '''
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[i] for i in range(size)], axis=0)   
    return result

def concat_patch(data, mesh=128, patch=32, size=8):
    '处理有重叠区域的预测结果'
    
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(mesh,mesh,mesh)
                # 处理重叠部分
                pred = pred[patch:-patch, patch:-patch, patch:-patch]
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[i] for i in range(size)], axis=0)   
    return result