#!python

import tensorflow as tf
from tensorflow import keras 

from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import tf_utils

## from https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/layers/merging/base_merge.py#L1

class DropConnect(keras.layers.Layer):
    def __init__(self, rate, data_type =tf.float32, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.rate = rate
        self.data_type = data_type

    def build(self, input_shape):
        shape = input_shape[-1] 
        # Sample Bernoulli Random Variable
        boolean = tf.random.uniform(shape =(1,), minval=0, maxval = 1) > self.rate
        # Make mask_matrix
        mask_matrix = tf.repeat(boolean, shape*shape)
        mask_matrix = tf.reshape(mask_matrix, (shape, shape))
        mask_matrix = tf.cast(mask_matrix, dtype = self.data_type)
        # Apply mask to the input
        self.StochasticDrop = keras.layers.Lambda(lambda x: tf.matmul(x, mask_matrix))

    def call(self, inputs, training = None):
        if training:
            outputs = self.StochasticDrop(inputs)
            return outputs
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_rate": self.rate}
        return {**base_config, **config}