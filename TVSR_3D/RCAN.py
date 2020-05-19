# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:42:54 2019

@author: gear
github:  https://github.com/gear106
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Activation, Add, Lambda, GlobalAveragePooling3D
from tensorflow.keras.layers import Multiply, Dense, Reshape, Conv3DTranspose
from tensorflow.keras.models import Model

import sys
sys.setrecursionlimit(10000)


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def sub_pixel_conv3d(scale=2):
    return Conv3DTranspose(16, 3, 2, padding='same')


def upsample(input_tensor, filters):
    x = Conv3D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv3d(scale=2)(x)
    x = Activation('relu')(x)
    return x


def ca(input_tensor, filters, reduce=16):
    x = GlobalAveragePooling3D()(input_tensor)
    x = Reshape((1, 1, 1, filters))(x)
    x = Dense(filters/reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def rcab(input_tensor, filters, scale=0.1):
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    ca(x, filters)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def rg(input_tensor, filters, n_rcab=10):
    x = input_tensor
    for _ in range(n_rcab):
        x = rcab(x, filters)
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def rir(input_tensor, filters, n_rg=5):
    x = input_tensor
    for _ in range(n_rg):
        x = rg(x, filters=filters)
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def generator(input_shape, filters=64, n_sub_block=3):
    inputs = Input(shape=input_shape)

    x = x_1 = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = rir(x, filters=filters)
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv3D(filters=1, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)

#autoencoder = generator(input_shape=(8,8,8,1))
#autoencoder.summary()