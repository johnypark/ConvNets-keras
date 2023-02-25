2/25/2023

Testing wider layer ResNet with lower resolution.

ResNet-50-Wide with 2x larger layer takes 14.5 miutes to train one epoch.

Increasing to 60 epochs basically have same validation Top-1 Acc. (63%)

Training ResNet-50 in 224x224 resol 20 epochs yielded 63% on validation Top-1 Acc, with mLR = 0.1 and WD = 1e-4.

It takes 16 minutes to train one epoch.


2/21/2023

Training on ResNet-50 160x160 resol in 20 epochs with CLR followed by Leslie Smith yielded 49.85 on validation F1. Training F1:55.90. Note that it is not Top-1 acc, so it underestimates the acc. 

It took 5-6 minutes to train one epoch.

Found that activation layer should be placed after the skip connection in the Residual Blocks. Fixed the issue.

- Investigating on the initializers: Updated them to ResNet-RS settings.

1. Convolution Layer initialization: https://www.tensorflow.org/api_docs/python/tf/keras/initializers

Currently using default settings for convolution layer initiailzer. ResNet50 uses default, ResNet-RS usees VarianceScaling. He_Normal seems to be universal in other codes.

1.1. keras ResNet-RS uses VarianceScaling initilizer:

https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/applications/resnet_rs.py#L123

Settings:
```python
CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}
```
1.2. keras ResNet-50 uses default initilizer:

https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/resnet.py#L245

2. Batch Normalization settings

ResNet50:
epsilon = 1.001e-5,
axis = 3 or 1 
ResNet-RS:
momentum = 0.0,
epsilon = 1e-5,
axis = 3 if backend.image_data_format() == "channels_last" else 1



