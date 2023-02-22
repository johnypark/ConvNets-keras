__copyright__ = """
Building Blocks of ConvNets
Copyright (c) 2023 John Park
"""
### style adapted from TensorFlow authors

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


KERNEL_INIT = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    }} #from resnet-rs


def ConvBlock(filters,
              kernel_size,
              strides = 1,
              padding = "same",
              activation: str = "relu",
              kernel_initializer = {
                "class_name": "VarianceScaling",
                "config": {"scale": 2.0, "mode": "fan_out",
                           "distribution": "truncated_normal" }}, 
              bn_momentum = 0,
              bn_epsilon = 1e-5,
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
                                    kernel_size, kernel_size, filters),
                                kernel_initializer = kernel_initializer,
                                **kwargs
                                )(x)
        x = keras.layers.BatchNormalization( momentum = bn_momentum,
                                             epsilon = bn_epsilon,
            name = name +"_batch_norm")(x)
        if activation:
            x = keras.layers.Activation(activation, name = name +"_act")(x)
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
                              name = name + "_4",
                              activation = None                           
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
                              name = name + "_3",
                              activation = None                           
                              )(r)

        x = tf.keras.layers.Add()([skip_connection, r])
        x = keras.layers.Activation(activation, name = name +"_act")(x)
    
        return x

    return apply


def Inverted_BN_Block(in_channels, 
                      out_channels, 
                      expansion_factor, 
                      stride, 
                      linear = True,
                      use_se=True, 
                      se_ratio=12,
                      **kwargs):
  
    #use_shortcut = stride == 1 and in_channels <= channels
    in_channels = in_channels
    if linear:
        act_ftn = None
    else:
        act_ftn= 'relu6'
    def apply(inputs):
          x = inputs
          skip_connection = inputs
          if expansion_factor != 1:
              expand = in_channels * expansion_factor
              x = ConvBlock(filters = expand, 
                        kernel_size = 1, 
                        stride = 1,
                        activation = "silu")(x)
          else:
              expand = in_channels

          x = ConvBlock(filters = expand,
                        kernel_size = 3,
                        stride = stride,
                        groups = expand,
                        activation = act_ftn, **kwargs)(x) # Double check the padding here!
                        # what is pytorch padding = 1 for keras???
                        # Look for padding = 1 in torch!! documents!
          #if use_se:
              # implement SE layer 
          x = keras.layers.Activation(act_ftn)(x)
          x = ConvBlock(filters = out_channels,
                        kernel_size = 1,
                        strides = 1,
                        activation = act_ftn)(x) 
          x = tf.keras.layers.Add()([skip_connection, x])
          # add activation layer here? 
          return x
    return apply


def SqueezeBlock(channels,
                 activaiton = 'relu'):
    def apply(inputs):
        x = inputs
        skip_connection = inputs
        x = ConvBlock(channels//2, kernel_size=1,
                  activation = activaiton)(x)
        x = ConvBlock(channels//4, kernel_size=1,
                  activation = activaiton)(x)
        x = ConvBlock(channels//2, kernel_size= (3,1),
                  activation = activaiton)(x)
        x = ConvBlock(channels//2, kernel_size= (1,3),
                  activation = activaiton)(x)
        x = ConvBlock(channels, kernel_size= (1,1),
                  activation = activaiton)(x)
        output = tf.keras.layers.Add()([skip_connection, x])
        
        return output
    
    return apply