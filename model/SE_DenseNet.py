import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Concatenate, Multiply, AveragePooling2D, MaxPooling2D, Reshape, Input, Dropout
from keras.models import Model
"""
Creates a SE-DenseNet Model as defined in:
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. (2016). 
Densely Connected Convolutional Networks. 
arXiv preprint arXiv:1608.06993.
import from https://github.com/liuzhuang13/DenseNet
And
Jie Hu, Li Shen, Gang Sun. (2017)
Squeeze-and-Excitation Networks
Copyright (c) Yang Lu, 2017
"""

def conv_block(x, num_filters, stride= 1, drop_rate= 0.2, padding='same', name_prefix=''):
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(x)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1 = Conv2D(num_filters, kernel_size=1, strides=stride, padding=padding, name=name_prefix + 'conv1')(act1)

    bn2 = BatchNormalization(name=name_prefix + 'bn2')(conv1)
    act2 = Activation('relu', name=name_prefix + 'relu2')(bn2)
    conv2 = Conv2D(num_filters, kernel_size=3, strides=stride, padding=padding, name=name_prefix + 'conv2')(act2)
    drop= Dropout(drop_rate, name=name_prefix + 'drop')(conv2)
    return drop

def muti_conv_block(x, num_filters, stride=1, padding='same', name_prefix=''):
    num_filters= num_filters//3
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(x)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1 = Conv2D(num_filters, kernel_size=1, strides=stride, padding=padding, name=name_prefix + 'conv1')(act1)

    bn2 = BatchNormalization(name=name_prefix + 'bn2')(x)
    act2 = Activation('relu', name=name_prefix + 'relu2')(bn2)
    conv2 = Conv2D(num_filters, kernel_size=3, strides=stride, padding=padding, name=name_prefix + 'conv2')(act2)

    bn3 = BatchNormalization(name=name_prefix + 'bn3')(x)
    act3 = Activation('relu', name=name_prefix + 'relu3')(bn3)
    conv3 = Conv2D(num_filters, kernel_size=5, strides=stride, padding=padding, name=name_prefix + 'conv3')(act3)

    return Concatenate()([conv1, conv2, conv3])
def transition_block(x, num_filters, name_prefix=''):
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(x)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1 = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', name=name_prefix + 'conv1')(act1)
    avrg1 = AveragePooling2D(pool_size=(2, 2), strides=2, name=name_prefix + 'avrg1')(conv1)
    return avrg1

def se_dense_block(x, num_conv, num_filters, reduction=8, name_prefix=''):
    conv = x
    for i in range(num_conv):
        conv = conv_block(conv, num_filters, stride=1, name_prefix=name_prefix + f'conv_{i+1}_')
    
        se = GlobalAveragePooling2D(name=name_prefix + f'se_avgpool_{i+1}')(conv)
        se = Dense(num_filters // reduction, activation='relu', name=name_prefix + f'se_dense1_{i+1}')(se)
        se = Dense(num_filters, activation='sigmoid', name=name_prefix + f'se_dense2_{i+1}')(se)
        se = Reshape((1, 1, num_filters), name=name_prefix + f'se_reshape_{i+1}')(se)
        se = Multiply(name=name_prefix + f'se_multiply_{i+1}')([conv, se])
    
        conv = Concatenate(name=name_prefix + f'concat_{i+1}')([conv, se])
    
    return conv

def multi_se_dense_block(x, num_conv, num_filters, reduction=8, name_prefix=''):
    conv = x
    for i in range(num_conv):
        conv = muti_conv_block(conv, num_filters, stride=1, name_prefix=name_prefix + f'conv_{i+1}_')
    
        se = GlobalAveragePooling2D(name=name_prefix + f'se_avgpool_{i+1}')(conv)
        se = Dense(num_filters // reduction, activation='relu', name=name_prefix + f'se_dense1_{i+1}')(se)
        se = Dense(num_filters, activation='sigmoid', name=name_prefix + f'se_dense2_{i+1}')(se)
        se = Reshape((1, 1, num_filters), name=name_prefix + f'se_reshape_{i+1}')(se)
        se = Multiply(name=name_prefix + f'se_multiply_{i+1}')([conv, se])
    
        conv = Concatenate(name=name_prefix + f'concat_{i+1}')([conv, se])
    
    return conv

def SE_Dense_Model(input_shape=(128, 128, 1), num_blocks=[4, 4, 4, 4], num_classes=5):
    inputs = Input(shape=input_shape)
    
    ba1= BatchNormalization(name='ba1')(inputs)
    conv = Conv2D(64, kernel_size=7, strides=2, padding='same', name='conv1')(ba1)
    conv = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='maxpool1')(conv)
    num_channels = 64
    growth_rate = 32
    x = conv
    for i, num_conv in enumerate(num_blocks):
        x = se_dense_block(x, num_conv=num_conv, num_filters=growth_rate, kernel_size=3, name_prefix=f'denseblock{i+1}_')
        num_channels += num_conv * growth_rate
        if i != len(num_blocks) - 1:
            x = transition_block(x, num_filters=num_channels // 2,  name_prefix=f'transition{i+1}_')
            num_channels = num_channels // 2
    
    bn = BatchNormalization(name='bn')(x)
    act = Activation('relu', name='relu')(bn)
    avg_pool = GlobalAveragePooling2D(name='avgpool')(act)
    outputs = Dense(num_classes, activation='softmax', name='output')(avg_pool)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def SE_DenseMulti_Model(input_shape= (128, 128, 1), layers= (6, 12, 24, 16), num_classes=5):
    inputs = Input(shape=input_shape)
    
    conv = conv_block(inputs, 64, kernel_size=7, stride=2, padding=3, name_prefix='conv1_')
    conv = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='maxpool1')(conv)
    num_channels = 64
    growth_rate = 32
    x = conv
    for i, num_conv in enumerate(layers):
        x = multi_se_dense_block(x, num_conv=num_conv, num_filters=growth_rate, kernel_size=3, name_prefix=f'denseblock{i+1}_')
        num_channels += num_conv * growth_rate
        if i != len(layers) - 1:
            x = transition_block(x, num_filters=num_channels // 2,  name_prefix=f'transition{i+1}_')
            num_channels = num_channels // 2
    
    bn = BatchNormalization(name='bn')(x)
    act = Activation('relu', name='relu')(bn)
    avg_pool = GlobalAveragePooling2D(name='avgpool')(act)
    outputs = Dense(num_classes, activation='softmax', name='output')(avg_pool)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

