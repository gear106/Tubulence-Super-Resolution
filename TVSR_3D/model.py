# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:10:01 2019

@author: gear
github:  https://github.com/gear106
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Activation, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from EDSR import generator
#from FFTSR import generator
#from RCAN import generator

class TVSR():a
    def __init__(self, lr_size, channels, saver, is_training=False, learning_rate=1e-4, 
                 batch_size=16, epochs=100):
        self.lr_size = lr_size    # 低分辨率输入尺寸
        self.channels = channels  # 输入图像通道个数
        self.saver = saver        # 保存地址
        self.learning_rate = learning_rate  
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()
        self.call_back_list =[
                ModelCheckpoint(filepath=self.saver,
                                monitor='loss', save_best_only=True)]
    #                EarlyStopping(monitor='val_loss', patience=5),
    
    def build_model(self):
        shape = (self.lr_size, self.lr_size, self.lr_size, self.channels)  
        model = generator(input_shape=shape)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
     
        return model
    
    def loss(self, y_true, y_pred):
        '''
        自定义loss, 每个速度变量的mse和合速度的mse
        '''               
        def _sum_square(x):
            return K.sum(K.square(x), axis=-1)
        loss2 = K.mean(K.square(_sum_square(y_pred) - _sum_square(y_true)), axis=-1)
        
        return loss2
    
    
    def train(self, train_X, train_Y):
        history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, 
                                 verbose=1, callbacks=self.call_back_list, validation_split=0.1)
        if self.is_training:
            self.save()
        
        return history    
        
    def process(self, input_X):
        predicted = self.model.predict(input_X)
        
        return predicted
    
    def load(self):
        weight_file = self.saver
        model = self.build_model()
        model.load_weights(weight_file)
        
        return model
        
    def save(self):
        self.model.save_weights(self.saver)
        
    
        
    
        
    
    
        

