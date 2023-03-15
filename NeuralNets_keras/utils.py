#!python

import tensorflow as tf
from tensorflow import keras 

from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import tf_utils

## from https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/layers/merging/base_merge.py#L1

class DropConnect_builtin(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(DropConnect_builtin, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        num_shapes = len(input_shape)
        shape = (None,)+(1,)*(num_shapes-1)
        self.StochasticDrop = keras.layers.Dropout(self.rate, noise_shape = shape)
        
    def call(self, inputs, training = None):
        return self.StochasticDrop(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_rate": self.rate}
        return {**base_config, **config}
    
    
    
class DropConnect(keras.layers.Layer): # this does not behave well, why?
    def __init__(self, rate, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        shape = input_shape[-1] 
        # Sample Bernoulli Random Variable
        boolean = tf.random.uniform(shape =(1,), minval=0, maxval = 1) > self.rate
        # Make mask_matrix
        mask_matrix = tf.repeat(boolean, shape*shape)
        mask_matrix = tf.reshape(mask_matrix, (shape, shape))
        mask_matrix = tf.cast(mask_matrix, dtype = self.compute_dtype)
        # Apply mask to the input
        self.StochasticDrop = keras.layers.Lambda(lambda x: tf.matmul(x, mask_matrix))

    def call(self, inputs, training = None):
        if training:
            outputs = self.StochasticDrop(inputs)
            return outputs
        return self.rate*inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_rate": self.rate}
        return {**base_config, **config}
    