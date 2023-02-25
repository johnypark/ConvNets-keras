# CCT: Escaping the Big Data Paradigm with Compact Transformers
# Paper: https://arxiv.org/pdf/2104.05704.pdf
# CCT-L/PxT: 
# L transformer encoder layers 
# T-layer convolutional tokenizer with PxP kernel size.
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
settings['transformerLayers'] = 2
settings['epsilon'] = 1e-6
projection_dim = 128
settings['denseInitializer'] = 'glorot_uniform'
settings['heads'] = 2
settings['conv2DInitializer'] = 'he_normal'

def CCT_tokenizer( 
              strides, 
              kernel_size,
              pool_size,
              pooling_stride,
              kernel_initializer,
              activation,
              out_channels = [64, 128], 
              name = None,
              padding = 'same',
              use_bias = False,
              **kwargs):
  
  def apply(inputs):
    x = inputs
    num_conv_tokenizers = len(out_channels)
    for k in range(num_conv_tokenizers):
      x = keras.layers.Conv2D(
        activation = activation,
        filters = out_channels[k],
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
                      shape = (-1, tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[3]),
                      tensor = x)
    return x

  return apply

def MLP_block(num_hidden_channels,
              DropOut,
              activation = 'gelu',
              name = None):
    
    def apply(inputs):
        x = inputs
        for hidden_channel in num_hidden_channels:
            x = keras.layers.Conv2D(filter = hidden_channel,
                                    kernel_size = 1)(x)
            x = keras.layers.Activation(activation)(x)
            x = keras.layers.Dropout(dropout_rate = DropOut)(x)
        return x
    return apply

def SeqPool(num_classes, settings): # Learnable pooling layer. In the paper they tested static pooling methods but leanrable weighting is more effcient
    # because each embedded patch does not contain the same amount of entropy. Enables the model to apply weights to tokens with repsect to the relevance of their information
    
    def apply(inputs):
        x = inputs    
        x = tf.keras.layers.LayerNormalization(
            epsilon = settings['epsilon'],
            name = 'final_norm'
        )(x)
        
        x = tf.squeeze( # why squeeze???
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
                )(x),
                b = x, 
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
            units = num_classes,
            use_bias = True
        )(x)

        return output

    return apply
        

### CCT MODEL
def cct(classes, 
        input_shape = (None, None, 3),
        num_heads = 2,
        projection_dim = 128,
        L_num_transformer_layers = 7,
        P_patch_size = 3,
        T_num_tokenizer_layers = 2,
        settings = settings,
        positional_embedding = True):

    """ CCT-L/PxT: L transformer encoder layers and PxP patch size.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    """
    
    # Need to add tokenizer settings
    input = tf.keras.layers.Input(
		shape = input_shape, 
		name = 'input')
    
    x = input
    x = CCT_tokenizer(strides = 1, 
              kernel_size = P_patch_size,
              pool_size = P_patch_size,
              pooling_stride = (P_patch_size-1),
              kernel_initializer = settings['conv2DInitializer'],
              activation = 'relu',
              out_channels = [64, 128])(x)
    
    if positional_embedding:
        
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

    ### dpr = [x for x in np.linspace(0, settings['stochasticDepthRate'], settings['transformerLayers'])] ### calculate stochastic depth probabilities
	### transformer block layers
    for k in range(L_num_transformer_layers):
        
        att = tf.keras.layers.LayerNormalization(
			epsilon = settings['epsilon'],
			name = f"transformer_{k}_norm"
		)(x)
        
        att = keras.layers.MultiHeadAttention(
			num_heads = num_heads, 
            key_dim = projection_dim,
            dropout = 0.1,
			name = f"transformer_{k}_attention"
		)(att, att)
        x = tf.keras.layers.Add()([att, x])
        x = tf.keras.layers.LayerNormalization(epsilon = settings['epsilon'])(x)
        mlp_out = MLP_block( num_hidden_channels = [projection_dim, projection_dim],
                      DropOut = 0.1, 
			name = f"transformer_{k}_mlp"
		)(x)
		#if settings['stochasticDepth']:
		#	recoder = StochasticDepth(dpr[k])(recoder)
    
    x = tf.keras.layers.Add()([mlp_out, x])
  
    #### Sequence Pooling ####
    
    output = SeqPool(num_classes = classes,
                     settings = settings)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)