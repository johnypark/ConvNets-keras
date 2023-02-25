# CCT: Escaping the Big Data Paradigm with Compact Transformers
# Paper: https://arxiv.org/pdf/2104.05704.pdf
# CCT-L/PxP: L transformer encoder layers and PxP patch size.
# In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
# CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification

settings = dict()
#settings = 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

settings['positionalEmbedding'] = True
settings['std_embedding'] = 0.2
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['dropout'] = 0.1


def CCT_tokenizer(out_channels, 
              activation,
              kernel_initializer,
              kernel_size,
              strides, 
              pool_size,
              pooling_stride,
              name = None,
              padding = 'same',
              use_bias = False,
              **kwargs):
  
  def apply(inputs):
    x = inputs
    for k in range(0, len(out_channels)):
      x = keras.layers.Conv2D(
        activation = activation,
        filters = out_channels,
        kernel_initializer = kernel_initializer,
        kernel_size = kernel_size,
        name = name,
        padding = padding,
        strides = strides,
        use_bias = use_bias,
        **kwargs
      )(x)
      x = keras.layers.MaxPool2D(
        name = name+"maxpool_1",
        padding =  padding,
        pool_size = pool_size, 
        strides = pooling_stride 
      )(x)
    x =  tf.reshape(name = name+'reshape_1',
                      shape = (-1, x.shape.as_list()[1]*x.shape.as_list()[2], x.shape.as_list()[3]),
                      tensor = x)
    return x

  return apply

### CCT MODEL
def cct(settings):

    """ CCT-L/PxP: L transformer encoder layers and PxP patch size.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    """
    
	input = tf.keras.layers.Input(
		(None, None, 3), 
		name = 'input'
	)
	x = input
	x = CCT_tokenizer(x)
	if settings['positionalEmbedding']:
		embedding = tf.random.truncated_normal(
			shape = (x.shape.as_list()[1], x.shape.as_list()[2]),
			mean = 0.0,
			stddev = settings['std_embedding'],
			dtype = tf.dtypes.float32,
			seed = random.randint(settings['randomMin'],
                         settings['randomMax']),
			name = 'learnable_embedding'
		)
		x = tf.math.add(x, embedding) # maybe change this to layer add?
		x = tf.keras.layers.Dropout(settings['dropout'])(x)
	dpr = [x for x in np.linspace(0, settings['stochasticDepthRate'], settings['transformerLayers'])] ### calculate stochastic depth probabilities
	### transformer block layers
	for k in range(settings['transformerLayers']):
		attention = tf.keras.layers.LayerNormalization(
			epsilon = settings['epsilon'],
			name = f"transformer_{k}_norm"
		)(cct)
		attention = selfAttention(
			attention, 
			heads = settings['heads'], 
			name = f"transformer_{k}_attention"
		)
		cct = tf.keras.layers.Add()([attention, cct])
		recoder = tf.keras.layers.LayerNormalization(epsilon = settings['epsilon'])(cct)
		recoder = mlpEncode(
			recoder, 
			name = f"transformer_{k}_mlp"
		)
		if settings['stochasticDepth']:
			recoder = StochasticDepth(dpr[k])(recoder)
		cct = tf.keras.layers.Add()([recoder, cct])
	cct = tf.keras.layers.LayerNormalization(
		epsilon = settings['epsilon'],
		name = 'final_norm'
	)(cct)
	cct = tf.squeeze(
		axis = -2,
		input = tf.matmul(
			a = tf.keras.layers.Dense(
				activation = 'softmax',
				activity_regularizer = None,
				bias_constraint = None,
				bias_initializer = 'zeros',
				bias_regularizer = None,
				kernel_constraint = None,
				kernel_initializer = settings['denseInitializer'],
				kernel_regularizer = None,
				name = 'weight',
				units = 1,
				use_bias = True
			)(cct),
			b = cct, 
			name = 'apply_weight',
			transpose_a = True
		),
		name = 'squeeze'
	)
	output = tf.keras.layers.Dense(
		activation = None,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = 'zeros',
		bias_regularizer = None,
		kernel_constraint = None,
		kernel_initializer = settings['denseInitializer'],
		kernel_regularizer = None,
		name = 'output',
		units = settings['classes'],
		use_bias = True
	)(cct)
	return tf.keras.Model(inputs = input, outputs = output)