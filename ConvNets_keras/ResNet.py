__copyright__ = """
Copyright (c) 2023 John Park
"""
### style adapted from TensorFlow authors
###

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_addons as tfa
import json
from tensorflow import keras


def ConvBlock(filters,
              kernel_size,
              strides = 1,
              padding = "same",
              activation: str = "relu",
              name = None, 
              **kwargs):
    """ 
    
    ConvBlock: Base unit of ResNet. keras.layers.Conv2D + BN + activation layers.

    Args: Argument style inherits keras.layers.Conv2D
        filters (int): # of channels.
        kernel_size (int): kernel size.
        strides (int, optional): strides in the Conv2D operation . Defaults to 1.
        padding (str, optional): padding in the Conv2D operation. Defaults to "same".
        activation (str, optional): name of the activation function. keras.layers.Activation. Defaults to "relu".
        name (str, optional): name of the layer. Defaults to None.
        
    """
    if name is None: # adopted this structure from tf.kera
        counter = keras.backend.get_uid("conv_")
        name = f"conv_{counter}"
      
    def apply(inputs):
        x = inputs
        x = keras.layers.Conv2D(filters = filters,
                                kernel_size = kernel_size,
                                padding = padding,
                                strides = strides,
                                name = name + "_{}x{}conv_ch{}".format(
                                    kernel_size, kernel_size, filters)
                                )(x)
        x = keras.layers.BatchNormalization(name = name +"_batch_norm")(x)
        x = keras.layers.Activation(activation, name = name +"_act")(x)
        return x
    
    return apply


def input_stem(ResNetType = "C",
                kernel_choice = 5,
                channels = 64,
                activation: str = "relu",
                name = None):
    if name is None: # adopted this structure from tf.keras
        counter = keras.backend.get_uid("stem_")
        name = f"stem_{counter}"
        
    """
    input stem of ResNet
    Kernel Choice of Stem: 3, 5, or 7
    Type Choice: "C" or others
    input stem described in the bag of tricks paper: Type C replaces the 7x7Conv layer with three 3x3Conv layers.
  
    """
  
    def apply(inputs):
        x = inputs
        N_kernel = kernel_choice
        idx = 1
        if ResNetType == "C":
            N_Conv3x3 = kernel_choice//3
            N_kernel = 3 
            for i in range(N_Conv3x3):  
                x = ConvBlock(filters = channels//2,
                              kernel_size = 3,
                              #padding = "same",
                              strides = 1,
                              name = name + "_" + str(idx)
                              )(x)
                idx = idx +1 
        
        x = ConvBlock(filters = channels,
                              kernel_size = N_kernel,
                              padding = "same",
                              strides = 2,
                              name = name + "_" + str(idx)                             
                              )(x)
        return x
    
    return apply

def BN_Res_Block( target_channels,
                  BottleNeck_channels, 
                 ResNetType = "C",
                 padding = "same",
                 downsampling = False,
                 activation: str = "relu",
                 name = None):
    
    """
    BN_Res_Block: BottleNeck Residual Block. type A, B, and D
    """
    #if target_channels == BottleNeck_channels:
    #if name is None: # adopted this structure from tf.keras
    #    counter = keras.backend.get_uid("Residual_")
    #    name = f"Residual_{counter}"
    #else:  
    if name is None: # adopted this structure from tf.keras
        counter = keras.backend.get_uid("BN_Residual_")
        name = f"BN_Residual_{counter}"
    
    def apply(inputs):
        prev_channels = inputs.shape[-1]
    #print(inputs.shape[-1])
        r = inputs # r for residual
        skip_connection = inputs 
        DownSamplingStride = 1
        #if target_channels == BottleNeck_channels:
      
        if prev_channels != target_channels:
            DownSamplingStride = 2
            skip_connection = ConvBlock(filters = target_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = DownSamplingStride,
                              name = name + "_4"                           
                              )(skip_connection)
    
        r = ConvBlock(filters = BottleNeck_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = 1,
                              name = name + "_1"                           
                              )(r)
        r = ConvBlock(filters = BottleNeck_channels,
                              kernel_size = 3,
                              padding = padding,
                              strides = DownSamplingStride,
                              name = name + "_2"                           
                              )(r)
        r = ConvBlock(filters = target_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = 1,
                              name = name + "_3"                           
                              )(r)

        x = tf.keras.layers.Add()([skip_connection, r])
    
        return x

    return apply


def ResNet( classes = 1000,
                include_top = True,                 
                input_shape = (None, None, 3),
                N_filters = [256, 512, 1024, 2048],
                N_BottleNecks = {256: 64, 512:128, 1024:256, 2048:512},
                 N_blocks = {256:3, 512:4, 1024:23, 2048:3},
                 stem_channels = 64,
                 stem_kernel = 7,
                 ResNetType = "C",
                 pooling = None,
                 ):
    input_data = keras.Input(shape = input_shape)
    x = input_stem(ResNetType = ResNetType, 
                  kernel_choice = stem_kernel,
                  channels = stem_channels)(input_data)
    for target_ch in N_filters:
        for ii in range(N_blocks[target_ch]):
            x = BN_Res_Block(target_channels = target_ch,
                  BottleNeck_channels = N_BottleNecks[target_ch]
                  )(x)
    if include_top:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense( classes, activation = "softmax")(x)
    
    out = x
    model = keras.Model(input_data, out)
    return model