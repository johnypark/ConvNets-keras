__copyright__ = """
Building Blocks of NeuralNets
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
              use_bias = False,
              kernel_initializer = {
                "class_name": "VarianceScaling",
                "config": {"scale": 2.0, "mode": "fan_out",
                           "distribution": "truncated_normal" }}, 
              bn_momentum = 0.0,
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
                                use_bias = use_bias,
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

# MHSA layer 
# Adopted from: https://github.com/faustomorales/vit-keras/blob/master/vit_keras/utils.py
# Also learn: https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, *args, num_heads, output_weight = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.output_weight = output_weight

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads !=0:
          raise ValueError(
              f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
              )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = keras.layers.Dense(hidden_size, name = "query")
        self.key_dense = keras.layers.Dense(hidden_size, name = "key")
        self.value_dense = keras.layers.Dense(hidden_size, name = "value")
        self.combine_heads = keras.layers.Dense(hidden_size, name = "out")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b = True)
        dim_key = tf.cast(tf.shape(key)[-1], dtype = score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis = -1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
                      tensor = x, 
                      shape = (batch_size, -1, self.num_heads, self.projection_dim)
                      )
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm = [0, 2, 1, 3])
        concat_attention = tf.reshape(attention, 
                                      shape = (batch_size, -1, self.hidden_size)
                                      )
        output = self.combine_heads(concat_attention)
        
        if self.output_weight:
            output = output, weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
# Feed Forward Network (FFN)

def MLP_block(embedding_dim,
              mlp_ratio,
              DropOut,
              activation = 'gelu',
              name = None):
    
    def apply(inputs):
        x = inputs
        x = keras.layers.Dense(units = int(embedding_dim*mlp_ratio))(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(rate = DropOut)(x)
        x = keras.layers.Dense(units = embedding_dim)(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(rate = DropOut)(x)
        
        return x # here apply stochastic depth layer
    
    return apply

# Transformer Block

def Transformer_Block(num_layers, 
                      mlp_ratio,
                      num_heads,
                      projection_dims,
                      DropOut_rate = 0.1,
                      LayerNormEpsilon = 1e-6):
    def apply(inputs):
        
        x = inputs
        
        for Layer in range(num_layers):
            
            att = tf.keras.layers.LayerNormalization(
			epsilon = LayerNormEpsilon
		    )(x)
            att = MultiHeadSelfAttention(
			num_heads = num_heads
			)(att)
            x = tf.keras.layers.Add()([att, x])
            x = tf.keras.layers.Dropout(rate = DropOut_rate)(x)
            mlp = tf.keras.layers.LayerNormalization(
            epsilon = LayerNormEpsilon
            )(x)
            mlp = MLP_block(embedding_dim = projection_dims,
                            mlp_ratio = mlp_ratio,
                      DropOut = DropOut_rate 
		    )(mlp)
            x = tf.keras.layers.Add()([mlp, x]) 
            
            outputs = x
        
        return outputs
    return apply
    
# Positional embedding

def sinusodial_embedding(num_patches, embed_dim):
    
        """ sinusodial embedding in the attention is all you need paper 
        example:
        >> plt.imshow(sinusodial_embedding(100,120).numpy()[0], cmap='hot',aspect='auto')
        """
    
        def criss_cross(k):
            n_odd = k//2
            n_even = k - k//2
            # even columns go first 
            even = list(range(n_even))
            odd = list(range(n_even, k))
            ccl = []
            for i in range(k//2):
                ccl = ccl+ [even[i]]+ [odd[i]]
            if k//2 != k/2:
                ccl = ccl + [even[k//2]]
            return ccl
            
        embed = tf.cast(([[p / (10000 ** (2 * (i//2) / embed_dim)) for i in range(embed_dim)] for p in range(num_patches)]), tf.float32)
        even_col =  tf.sin(embed[:, 0::2])
        odd_col = tf.cos(embed[:, 1::2])
        embed = tf.concat([even_col, odd_col], axis = 1)
        embed = tf.gather(embed, criss_cross(embed_dim), axis = 1)
        embed = tf.expand_dims(embed, axis=0)

        return embed

class add_positional_embedding():
    
    def __init__(self, 
                 patch_length, 
                 embedding_dim,
                 embedding_type = 'sinusodial'):
        
        self.embedding_type = embedding_type
        self.patch_length = patch_length
        self.embedding_dim = embedding_dim
        if embedding_type:
            if embedding_type == 'sinusodial':
                self.positional_embedding = tf.Variable(sinusodial_embedding(num_patches = self.patch_length,
                                              embed_dim = self.embedding_dim
                                              ),
                    trainable = False)
            elif embedding_type == 'learnable':
                self.positional_embedding = tf.Variable(tf.random.truncated_normal(shape=[1, self.patch_length, self.embedding_dim], stddev=0.2))
            
        else:
            self.positional_embedding = None
        
    def __call__(self, input):
        input = tf.keras.layers.Add(name = 'add_positional_embedding')([input, self.positional_embedding]) # tf math add or concat? 
        return input