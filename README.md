# ConvNets-keras

Convolution Neural Networks re-implementation in keras

The purpose of this project is flexible options to customize modern CNNs in keras.

## Installation
 
```
 
 pip install git+https://github.com/johnypark/ConvNets-keras@main

```

## ResNet

ResNet() allows customizing number of channels, bottleneck layers, and number of blocks. 


## [ReXNet](https://github.com/johnypark/ConvNets-keras/blob/main/ConvNets_keras/ReXNet.py)
Rethinking Channel Dimensions for Efficient Model Design 

https://arxiv.org/pdf/2007.00992.pdf

Official pytorch implementation: https://github.com/clovaai/rexnet



ResNet() allows customizing number of channels, bottleneck layers, and number of blocks. 


### Example

Usage example building ResNet-50 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XlDZWoYzNMYNRZnCsTA6exesbn_P85nF?usp=sharing)

Result: 
```
 BN_Residual_16_2_batch_norm (B  (None, 7, 7, 512)   2048        ['BN_Residual_16_2_3x3conv_ch512[
 atchNormalization)                                              0][0]']                          
                                                                                                  
 BN_Residual_16_2_act (Activati  (None, 7, 7, 512)   0           ['BN_Residual_16_2_batch_norm[0][
 on)                                                             0]']                             
                                                                                                  
 BN_Residual_16_3_1x1conv_ch204  (None, 7, 7, 2048)  1050624     ['BN_Residual_16_2_act[0][0]']   
 8 (Conv2D)                                                                                       
                                                                                                  
 BN_Residual_16_3_batch_norm (B  (None, 7, 7, 2048)  8192        ['BN_Residual_16_3_1x1conv_ch2048
 atchNormalization)                                              [0][0]']                         
                                                                                                  
 BN_Residual_16_3_act (Activati  (None, 7, 7, 2048)  0           ['BN_Residual_16_3_batch_norm[0][
 on)                                                             0]']                             
                                                                                                  
 add_15 (Add)                   (None, 7, 7, 2048)   0           ['add_14[0][0]',                 
                                                                  'BN_Residual_16_3_act[0][0]']   
                                                                                                  
 global_average_pooling2d (Glob  (None, 2048)        0           ['add_15[0][0]']                 
 alAveragePooling2D)                                                                              
                                                                                                  
 dense (Dense)                  (None, 1000)         2049000     ['global_average_pooling2d[0][0]'
                                                                 ]                                
                                                                                                  
==================================================================================================
Total params: 25,656,136
Trainable params: 25,602,888
Non-trainable params: 53,248
__________________________________________________________________________________________________
```
 
