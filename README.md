# ConvNets-keras

Practical Re-Implementation of keras

The purpose of this project is to offer flexible customizable options for modern CNNs in keras.

Reference Papers:
1. Residual Blocks and BottleNeck Structure:
- [Deep Residual Learning for Image Recognition (Kaiming He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks (Tong He et al., 2018)](https://arxiv.org/abs/1812.01187)
- [Identity Mappings in Deep Residual Networks (Kaiming He et al., 2016)](https://arxiv.org/abs/1603.05027)
- [ResNeXt: Aggregated Residual Transformations for Deep Neural Networks (Saining Xie et al., 2016)](https://arxiv.org/abs/1611.05431)
- [ReXNet: Rethinking Channel Dimensions for Efficient Model Design (Dongyoon Han et al., 2021)](https://arxiv.org/abs/2007.00992)
- [ResNet strikes back: An improved training procedure in timm (Ross Wightman et al., 2021)](https://arxiv.org/abs/2110.00476)
- [Revisiting ResNets: Improved Training and Scaling Strategies (Irwan Bello et al., 2021)](https://arxiv.org/abs/2103.07579)

2. Inverted Residual Blocks and Linear BottleNeck
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Mark Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

3. Other structures
- [SqueezeNext: Hardware-Aware Neural Network Design](https://arxiv.org/abs/1803.10615)
- [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

## Installation
 
```
 
 pip install git+https://github.com/johnypark/ConvNets-keras@main

```

## ResNet

ResNet() allows customizing number of channels, bottleneck layers, and number of blocks. 


## [ReXNet](https://github.com/johnypark/ConvNets-keras/blob/main/ConvNets_keras/ReXNet.py)

Official pytorch implementation: https://github.com/clovaai/rexnet

ResNet() allows customizing number of channels, bottleneck layers, and number of blocks. 

## Building Blocks ##

1. ConvBlock: Basic convolutional layer followed by batch normalization and activaiton function.

2. BN_Res_Block: Building unit of ResNet, with BottleNeck structure first descirbed in He et al., (2015).  

3. Inverted_BN_Block: Building unit of ReXNet, with a modified version of inverted BottleNeck structure described in Han et al. (2021), originally invented in Snadler et al. (2018).

### Example

Usage example building ResNet-50 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XlDZWoYzNMYNRZnCsTA6exesbn_P85nF?usp=sharing)

``` python
import ConvNets_keras as cnn

rs50 = cnn.ResNet(classes = 1000,
                input_shape = (224, 224, 3),
                N_filters = [256, 512, 1024, 2048],  
                N_BottleNecks = {256: 64, 512:128, 1024:256, 2048:512},
                N_blocks = {256:3, 512:4, 1024:6, 2048:3},
                stem_channels = 64,
                stem_kernel = 7,
                ResNetType = "C",
                pooling = "average",
                 )
```
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
 
