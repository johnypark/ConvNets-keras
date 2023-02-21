__copyright__ = """
ReXNet keras implementation
Copyright (c) 2023 John Park
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
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
import tensorflow as tf
from tensorflow import keras
from ConvNets_keras.building_blocks import *


def RexNetV1(input_ch=16, 
             final_ch=180, 
             width_mult=1.0, 
             depth_mult=1.0, 
             classes=1000,
             use_se=True,
             se_ratio=12,
             dropout_ratio=0.2,
             bn_momentum=0.9):
    
        from math import ceil
      
        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []
        def apply(inputs):
            x = inputs
            # The following channel configuration is a simple instance to make each layer become an expand layer.
            for i in range(depth // 3):
                if i == 0:
                    in_channels_group.append(int(round(stem_channel * width_mult)))
                    channels_group.append(int(round(inplanes * width_mult)))
                else:
                    in_channels_group.append(int(round(inplanes * width_mult)))
                    inplanes += final_ch / (depth // 3 * 1.0)
                    channels_group.append(int(round(inplanes * width_mult)))
            
            x = ConvBlock(filters = int(round(stem_channel * width_mult)),
                  kernel_size = 3,
                  strides = 2,
                  padding = 'same', # in pytorch padding = 1 ?
                  activation = 'silu'
                  )(x)
     

            for idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
                x = Inverted_BN_Block(in_channels=in_c,
                                             out_channels=c,
                                             expansion_factor=t,
                                             stride=s,
                                             use_se=se, 
                                              se_ratio=se_ratio)(x)
            
            pen_channels = int(1280 * width_mult)
            x = ConvBlock(filters = pen_channels,
                  kernel_size = 1,
                  strides = 1,
                  padding = 'same', # in pytorch padding = 1 ?
                  activation = 'silu'
                  )(x)
            x = keras.layers.GlobalAveragePooling2D()
            x = keras.layers.Dropout(dropout_ratio)
            x = keras.layers.Dense(classes, activation = 'softmax')
            
            return x
        return apply
