# CCT: Escaping the Big Data Paradigm with Compact Transformers
# Paper: https://arxiv.org/pdf/2104.05704.pdf
# CCT-L/KxT: 
# K transformer encoder layers 
# T-layer convolutional tokenizer with KxK kernel size.
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
settings['denseInitializer'] = 'glorot_uniform'
settings['heads'] = 2
settings['conv2DInitializer'] = 'he_normal'

def Conv_Tokenizer(
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
                      shape = (-1, tf.shape(x)[3], tf.shape(x)[1]*tf.shape(x)[2]),
                      tensor = x)
    return x

  return apply


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




def get_dim_Conv_Tokenizer(Conv_strides, pool_strides, num_tokenizer_ConvLayers):

  def apply(input):

    start = input
    for k in range(num_tokenizer_ConvLayers):
      Conv_out = -(start // -Conv_strides)
      pool_out = -(Conv_out // - pool_strides)
      start = pool_out
    
    return pool_out
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
        

### CCT MODEL
def CCT(classes, 
        input_shape = (None, None, 3),
        num_TransformerLayers = 14,
        num_heads = 6,
        mlp_ratio = 3,
        embedding_dim = 384,
        tokenizer_kernel_size = 7,
        tokenizer_strides = 2,
        num_tokenizer_ConvLayers = 2,
        DropOut_rate = 0.1,
        settings = settings,
        positional_embedding = True):

    """ CCT-L/PxT: L transformer encoder layers and PxP patch size.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    """
    Tokenizer_ConvLayers_dims = [embedding_dim//2**(i) for i in reversed(range(num_tokenizer_ConvLayers))]
    # Need to add tokenizer settings
    input = tf.keras.layers.Input(
		shape = input_shape)
    
    x = input
    x = Conv_Tokenizer(strides = tokenizer_strides, 
              kernel_size = tokenizer_kernel_size,
              #kernel_initializer = settings['conv2DInitializer'],
              activation = 'relu',
              pool_size = 3,
              pooling_stride = 2,
              list_embedding_dims = Tokenizer_ConvLayers_dims)(x)
    
    if positional_embedding: # this does not work!
        
        embedding = tf.random.truncated_normal(
			shape = (tf.shape(x)[1], tf.shape(x)[2]),
			mean = 0.0,
			stddev = settings['std_embedding'],
			dtype = tf.dtypes.float32,
			seed = random.randint(settings['randomMin'],
                         settings['randomMax']),
			name = 'learnable_embedding'
		)
        
        x = tf.math.add(x, embedding) # maybe change this to layer add?
    x = tf.keras.layers.Dropout(settings['dropout'])(x)
    projection_dims = get_dim_Conv_Tokenizer(Conv_strides = tokenizer_strides, 
                                             pool_strides = 2, 
                                             num_tokenizer_ConvLayers = num_tokenizer_ConvLayers)(input_shape[0])
    projection_dims = projection_dims**2
    ### dpr = [x for x in np.linspace(0, settings['stochasticDepthRate'], settings['transformerLayers'])] ### calculate stochastic depth probabilities
	### transformer block layers
    for k in range(num_TransformerLayers):
        
        att = tf.keras.layers.LayerNormalization(
			epsilon = settings['epsilon'],
			#name = f"transformer_{k}_norm"
		)(x)
        
        att = keras.layers.MultiHeadAttention(
			num_heads = num_heads, 
            key_dim = projection_dims,
            dropout = DropOut_rate,
            attention_axes = 1
			#name = f"transformer_{k}_attention"
		)(att, att)
        x = tf.keras.layers.Add()([att, x])
        x = tf.reshape(shape = (-1, tf.shape(x)[2], tf.shape(x)[1]),
                      tensor = x)
        x = tf.keras.layers.LayerNormalization(epsilon = settings['epsilon'])(x)
        mlp_out = MLP_block(embedding_dim = embedding_dim,
                            mlp_ratio = mlp_ratio,
                      DropOut = DropOut_rate 
		)(x)
        x = tf.keras.layers.Add()([mlp_out, x]) # do a stochastic depth layer here 
        x = tf.reshape(shape = (-1, tf.shape(x)[2], tf.shape(x)[1]),
                      tensor = x)

    #### Sequence Pooling ####
    
    output = SeqPool(num_classes = classes,
                     settings = settings)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)



### CCT MODEL
def CCTV2(classes, 
        input_shape = (None, None, 3),
        num_TransformerLayers = 14,
        num_heads = 6,
        mlp_ratio = 3,
        embedding_dim = 384,
        tokenizer_kernel_size = 7,
        tokenizer_strides = 2,
        num_tokenizer_ConvLayers = 2,
        DropOut_rate = 0.1,
        settings = settings,
        positional_embedding = True):

    """ CCT-L/PxT: L transformer encoder layers and PxP patch size.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    """
    Tokenizer_ConvLayers_dims = [embedding_dim//2**(i) for i in reversed(range(num_tokenizer_ConvLayers))]
    # Need to add tokenizer settings
    input = tf.keras.layers.Input(
		shape = input_shape)
    
    x = input
    x = Conv_TokenizerV2(strides = tokenizer_strides, 
              kernel_size = tokenizer_kernel_size,
              #kernel_initializer = settings['conv2DInitializer'],
              activation = 'relu',
              pool_size = 3,
              pooling_stride = 2,
              list_embedding_dims = Tokenizer_ConvLayers_dims)(x)
    
    if positional_embedding: # this does not work!
        
        embedding = tf.random.truncated_normal(
			shape = (tf.shape(x)[1], tf.shape(x)[2]),
			mean = 0.0,
			stddev = settings['std_embedding'],
			dtype = tf.dtypes.float32,
			seed = random.randint(settings['randomMin'],
                         settings['randomMax']),
			name = 'learnable_embedding'
		)
        
        x = tf.math.add(x, embedding) # maybe change this to layer add?
    x = tf.keras.layers.Dropout(settings['dropout'])(x)
    #projection_dims = get_dim_Conv_Tokenizer(Conv_strides = tokenizer_strides, 
    #                                         pool_strides = 2, 
    #                                         num_tokenizer_ConvLayers = num_tokenizer_ConvLayers)(input_shape[0])
    projection_dims = embedding_dim
    ### dpr = [x for x in np.linspace(0, settings['stochasticDepthRate'], settings['transformerLayers'])] ### calculate stochastic depth probabilities
	### transformer block layers
    for k in range(num_TransformerLayers):
        
        att = tf.keras.layers.LayerNormalization(
			epsilon = settings['epsilon'],
			#name = f"transformer_{k}_norm"
		)(x)
        
        att = keras.layers.MultiHeadAttention(
			num_heads = num_heads, 
            key_dim = projection_dims,
            dropout = DropOut_rate,
			#name = f"transformer_{k}_attention"
		)(att, att)
        x = tf.keras.layers.Add()([att, x])
        x = tf.keras.layers.LayerNormalization(epsilon = settings['epsilon'])(x)
        mlp_out = MLP_block(embedding_dim = projection_dims,
                            mlp_ratio = mlp_ratio,
                      DropOut = DropOut_rate 
		)(x)
        x = tf.keras.layers.Add()([mlp_out, x]) # do a stochastic depth layer here 

    #### Sequence Pooling ####
    
    output = SeqPool(num_classes = classes,
                     settings = settings)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)