# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:48:53 2019

@author: gear
github:  https://github.com/gear106
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda
from tensorflow.keras.models import Model


def res_block(input_tensor, filters, scale=0.1):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def sub_pixel_Conv2D(scale=2, **kwargs):
    ''' 
    注：tensorflow1.x版本为tf.depth_to_space
        tensorflow2.x版本为tf.nn.depth_to_space
        此函数只适用于4D tensor.
    '''
    
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)



def upsample(input_tensor, filters):
    '上采样函数，单次上采样2x'
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_Conv2D(scale=2)(x)
    x = Activation('relu')(x)
    return x


def generator(input_shape, filters=128, n_id_block=8, n_sub_block=2):
    inputs = Input(shape=input_shape)

    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)

    for _ in range(n_id_block):
        x = res_block(x, filters=filters)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)