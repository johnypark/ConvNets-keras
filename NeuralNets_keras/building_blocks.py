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

# modified from tensorflow addons https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/layers/stochastic_depth.py#L5-L90
class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, survival_probability: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.survival_probability = survival_probability

    def call(self, x, training=None):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x

        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_out = keras.backend.random_bernoulli(
            [], p=self.survival_probability
        )

        def _call_train():
            return tf.keras.layers.Add()(shortcut, b_out * residual)

        def _call_test():
            return tf.keras.layers.Add()(shortcut, residual)

        return tf.keras.backend.in_train_phase(
            _call_train, _call_test, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()

        config = {"survival_probability": self.survival_probability}

        return {**base_config, **config}
    
    
def Conv_TokenizerV2(
              kernel_size,
              strides = 2, 
              ##kernel_initializer,
              activation = 'relu',
              list_embedding_dims = [256], 
              pool_size = 3,
              pooling_stride = 2,
              name = None,
              padding = 'same',
              use_bias = False,
              **kwargs):
  
  def apply(inputs):
    #strides = strides if strides is not None else max(1, (kernel_size // 2) - 1)
    #padding = padding if padding is not None else max(1, (kernel_size // 2))
    
    x = inputs
    num_conv_tokenizers = len(list_embedding_dims)
    for k in range(num_conv_tokenizers):
      x = keras.layers.Conv2D(
        activation = activation,
        filters = list_embedding_dims[k],
        kernel_size = kernel_size,
        strides = strides,
        #kernel_initializer = kernel_initializer,
        name = name,
        padding = padding,
        use_bias = use_bias,
        **kwargs
      )(x)
      x = keras.layers.MaxPool2D(
        #name = name+"maxpool_1",
        pool_size = pool_size, 
        strides = pooling_stride,
        padding = padding
      )(x)
    x =  tf.reshape(#name = name+'reshape_1',
                      shape = (-1, tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[3]),
                      tensor = x)
    return x

  return apply


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
                self.positional_embeding = tf.Variable(sinusodial_embedding(num_patches = self.patch_length,
                                              embed_dim = self.embedding_dim
                                              ),
                    trainable = False)
            elif embedding_type == 'learnable':
                self.positional_embedding = tf.Variable(tf.random.truncated_normal(shape=[1, self.patch_length, self.embedding_dim], stddev=0.2))
            
        else:
            self.positional_emb = None
        
    def __call__(self, input):
        input = tf.keras.layers.Add(name = 'add_positional_embedding')([input, self.positional_embedding]) # tf math add or concat? 
        return input
        
        
def SeqPool(num_classes, settings): # Learnable pooling layer. In the paper they tested static pooling methods but leanrable weighting is more effcient
    # because each embedded patch does not contain the same amount of entropy. Enables the model to apply weights to tokens with repsect to the relevance of their information
    
    def apply(inputs):
        x = inputs    
        x = tf.keras.layers.LayerNormalization(
            epsilon = settings['epsilon'],
            name = 'final_norm'
        )(x)
        x_init = x
        x = tf.keras.layers.Dense(units = 1, activation = 'softmax')(x)
        x = tf.transpose(x, perm = [0, 2, 1])
        x = tf.matmul(x, x_init)
        x = tf.squeeze(x, axis = 1)     
        output = tf.keras.layers.Dense(
            activation = None,
            activity_regularizer = None,
            bias_constraint = None,
            bias_initializer = 'zeros',
            bias_regularizer = None,
            kernel_constraint = None,
            kernel_initializer = settings['denseInitializer'],
            kernel_regularizer = None,
            #name = 'output',
            units = num_classes,
            use_bias = True
        )(x)

        return output

    return apply


class extract_by_size():
    def __init__(self, patch_size, padding = 'VALID'):
        self.patch_size = patch_size
        self.padding = padding
        
    def __call__(self, input):
        x = tf.image.extract_patches( images = input, 
                                  sizes = [1, self.patch_size, self.patch_size, 1],
                                  strides = [1, self.patch_size, self.patch_size, 1],
                                  rates = [1, 1, 1, 1],
                                  padding = self.padding
                                  )
        return x


class extract_by_patch():
  def __init__(self, n_patches, padding = 'VALID'):
    self.n_patches = n_patches
    self.padding = padding

  def get_overlap(self, image_size, n_patches):
    from math import ceil
    n_overlap = n_patches - 1
    patch_size = ceil(image_size/ n_patches)
    return patch_size, (n_patches*patch_size - image_size) // n_overlap
    
  
  def __call__(self, inputs):
    patch_size_x, overlap_x = self.get_overlap(image_size = tf.shape(inputs)[1], n_patches = self.n_patches )
    patch_size_y, overlap_y = self.get_overlap(image_size = tf.shape(inputs)[2], n_patches = self.n_patches )
    
    result = tf.image.extract_patches(images = inputs,
                           sizes=[1, patch_size_x, patch_size_y, 1],
                           strides=[1, (patch_size_x - overlap_x), (patch_size_y - overlap_y), 1],
                           rates=[1, 1, 1, 1],
                           padding= self.padding)

    return result
