#!python

import tensorflow as tf
from tensorflow import keras 

# modified from tensorflow addons https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/layers/stochastic_depth.py#L5-L90
class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, survival_probability: float = 0.9, **kwargs):
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