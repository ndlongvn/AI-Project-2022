import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Concatenate, Multiply, AveragePooling2D, MaxPooling2D, Reshape, Input, Dropout
from keras.models import Model
import keras
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

# se + densenet
def se_dense_bottleneck(x, expansion=4, growthRate=32, dropRate=0.1, name_prefix=''):
    planes = expansion * growthRate
    bn1= BatchNormalization(name=name_prefix + 'bn1')(x)
    act1= Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1= Conv2D(planes, kernel_size=1, padding='same', name=name_prefix + 'conv1', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(act1)
    
    #
    outplanes= x.shape[-1] + growthRate
    # outplanes= growthRate
    bn2= BatchNormalization(name=name_prefix + 'bn2')(conv1)
    conv2= Conv2D(growthRate, kernel_size=3, padding='same', name=name_prefix + 'conv2', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(bn2)
    
    if dropRate > 0:
        conv2= Dropout(dropRate, name=name_prefix + 'drop')(conv2)

    conv2= Concatenate(axis=-1, name=name_prefix + 'concat')([x, conv2])
    se= GlobalAveragePooling2D(name=name_prefix + 'gap')(conv2)
    se= Dense(outplanes//8, activation='relu', name=name_prefix + 'fc1', kernel_initializer="he_uniform")(se)
    se= Dense(outplanes, activation='sigmoid', name=name_prefix + 'fc2', kernel_initializer="he_uniform")(se)
    se= Reshape((1, 1, outplanes), name=name_prefix + 'reshape')(se)
    se= Multiply(name=name_prefix + 'multiply')([conv2, se])
    return se

def transition_block(x, outplanes, name_prefix=''):
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(x)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1 = Conv2D(outplanes, kernel_size=1, strides=1, padding='same', name=name_prefix + 'conv1', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(act1)
    avrg1 = AveragePooling2D(pool_size=(2, 2), strides=2, name=name_prefix + 'avrg1')(conv1)
    return avrg1

def make_layer(x, layer, growthRate, dropRate=0.1, name_prefix=''):
    for i in range(layer):
        x= se_dense_bottleneck(x, growthRate= growthRate, dropRate= dropRate, name_prefix= name_prefix + f'block_{i+1}_')
    return x
def make_transition(x, compressionRate, name_prefix=''):
    outplanes= int(x.shape[-1] * compressionRate)
    x= transition_block(x, compressionRate, name_prefix= name_prefix + 'transition_')
    return x

def se_densenet(input_shape, growthRate= 32, head7x7= True, dropRate= 0, increasingRate= 1, compressionRate= 2, layers= [6, 12, 24, 16], num_classes= 5):
    headplanes= growthRate * pow(increasingRate, 2)
    input= Input(shape=input_shape, name='input')
    if head7x7:
        head= Conv2D(headplanes*2, kernel_size=7, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)
    else:
        head= Conv2D(headplanes, kernel_size=3, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)

        head= Conv2D(headplanes, kernel_size=3, strides=1, padding='same', name='head_conv2', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn2')(head)
        head= Activation('relu', name='head_relu2')(head)

        head= Conv2D(headplanes*2, kernel_size=3, strides=1, padding='same', name='head_conv3', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn3')(head)
        head= Activation('relu', name='head_relu3')(head)
    head= MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='head_pool')(head)

    for i, layer in enumerate(layers):
        x= make_layer(head, layer, growthRate, dropRate= dropRate, name_prefix= f'block{i+1}_')
        head= make_transition(x, compressionRate, name_prefix= f'transition{i+1}_')

    head= BatchNormalization(name='head_bn4')(head)
    head= Activation('relu', name='head_relu4')(head)
    head= GlobalAveragePooling2D(name='head_gap')(head)
    head= Dense(num_classes, activation='softmax', name='head_fc')(head)
    model= Model(inputs=input, outputs=head)
    return model

def SE_DenseNet121(input_shape, num_classes=5):
    return se_densenet(input_shape, layers=[6, 12, 24, 16], num_classes=num_classes)

def SE_DenseNet121_g64(input_shape, growRate=64, num_classes=5):
    return se_densenet(input_shape, layers=[6, 12, 24, 16], growthRate=growRate, num_classes=num_classes)




def se_dense_multi_bottleneck(x, expansion=4, growthRate=32, dropRate=0, name_prefix=''):
    planes= expansion * growthRate
    input= x
    output= x
    outplanes= x.shape[-1] + growthRate
    for kernel1, kernel2 in ((1, 3), (3, 5), (5, 7)):
        bn1= BatchNormalization(name=name_prefix + f'bn1_{kernel1}_{kernel2}')(input)
        act1= Activation('relu', name=name_prefix + f'relu1_{kernel1}_{kernel2}')(bn1)
        conv1= Conv2D(planes, kernel_size=kernel1, padding='same', name=name_prefix + f'conv1_{kernel1}_{kernel2}', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(act1)

        bn2= BatchNormalization(name=name_prefix + f'bn2_{kernel1}_{kernel2}')(conv1)
        act2= Activation('relu', name=name_prefix + f'relu2_{kernel1}_{kernel2}')(bn2)
        conv2= Conv2D(growthRate, kernel_size=kernel2, padding='same', name=name_prefix + f'conv2_{kernel1}_{kernel2}', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(act2)

        if dropRate > 0:
            conv2= Dropout(dropRate, name=name_prefix + 'drop')(conv2)
        conv2= Concatenate(name=name_prefix + f'concat_{kernel1}_{kernel2}')([x, conv2])
        se= GlobalAveragePooling2D(name=name_prefix + 'gap')(conv2)
        se= Dense(outplanes//8, activation='relu', name=name_prefix + f'fc1_{kernel1}_{kernel2}', kernel_initializer="he_uniform")(se)
        se= Dense(outplanes, activation='sigmoid', name=name_prefix + f'fc2_{kernel1}_{kernel2}', kernel_initializer="he_uniform")(se)
        se= Reshape((1, 1, outplanes), name=name_prefix + f'reshape_{kernel1}_{kernel2}')(se)
        se= Multiply(name=name_prefix + f'multi_{kernel1}_{kernel2}')([conv2, se])
        output= Concatenate(name=name_prefix + f'concat_{kernel1}_{kernel2}')([output, se])
    
    output= Dense(outplanes, activation='relu', name=name_prefix + 'fc3', kernel_initializer="he_uniform")(output)

    return output   
    
def make_layer_multi(x, layer, growthRate, dropRate=0, name_prefix=''):
    for i in range(layer):
        x= se_dense_multi_bottleneck(x, growthRate= growthRate, dropRate= dropRate, name_prefix= name_prefix + f'block_{i+1}_')
    return x

def se_densenet_multi(input_shape, growthRate= 32, head7x7= True, dropRate= 0, increasingRate= 1, compressionRate= 2, layers= [6, 12, 24, 16], num_classes= 5):
    headplanes= growthRate * pow(increasingRate, 2)
    input= Input(shape=input_shape, name='input')
    if head7x7:
        head= Conv2D(headplanes*2, kernel_size=7, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)
    else:
        head= Conv2D(headplanes, kernel_size=3, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)

        head= Conv2D(headplanes, kernel_size=3, strides=1, padding='same', name='head_conv2', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn2')(head)
        head= Activation('relu', name='head_relu2')(head)

        head= Conv2D(headplanes*2, kernel_size=3, strides=1, padding='same', name='head_conv3', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn3')(head)
        head= Activation('relu', name='head_relu3')(head)
    head= MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='head_pool')(head)

    for i, layer in enumerate(layers):
        x= make_layer_multi(head, layer, growthRate, dropRate= dropRate, name_prefix= f'block{i+1}_')
        head= make_transition(x, compressionRate, name_prefix= f'transition{i+1}_')

    head= BatchNormalization(name='head_bn4')(head)
    head= Activation('relu', name='head_relu4')(head)
    head= GlobalAveragePooling2D(name='head_gap')(head)
    head= Dense(num_classes, activation='softmax', name='head_fc')(head)
    model= Model(inputs=input, outputs=head)
    return model

def SE_DenseNet121_Multi(input_shape, num_classes=5):
    return se_densenet_multi(input_shape, layers=[6, 12, 24, 16], num_classes=num_classes)

def SE_DenseNet121_Multi_g64(input_shape, growRate=64, num_classes=5):
    return se_densenet(input_shape, layers=[6, 12, 24, 16], growthRate=growRate, num_classes=num_classes)


def transition_block(x, outplanes, padding= 'same', name_prefix=''):
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(x)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    conv1 = Conv2D(outplanes, kernel_size=1, strides=1, padding=padding, name=name_prefix + 'conv1', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(act1)
    avrg1 = AveragePooling2D(pool_size=(2, 2), strides=2, name=name_prefix + 'avrg1')(conv1)
    return avrg1

def cnn_block(x, outplanes, kernel_size, stride, padding, name_prefix=''):
    conv1 = Conv2D(outplanes, kernel_size=kernel_size, strides=stride, padding=padding, name=name_prefix + 'conv1', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    bn1 = BatchNormalization(name=name_prefix + 'bn1')(conv1)
    act1 = Activation('relu', name=name_prefix + 'relu1')(bn1)
    return act1
def se_layer(x, reduction=8, name_prefix=''):
    channel= x.shape[-1]
    se= GlobalAveragePooling2D(name=name_prefix + 'gap')(x)
    se= Dense(channel//reduction, activation='relu', name=name_prefix+ 'ds1')(se)
    se= Dense(channel, activation='sigmoid', name= name_prefix+ 'ds2')(se)
    se= Reshape((1, 1, channel), name= name_prefix+ 'rs')(se)
    se= Multiply(name= name_prefix+ 'multi')([x, se])
    return se

def multi_branch_cnn(x, n_output, kernel_size, stride, padding, drop_out=0.2, name_prefix=''):
    out1= []
    # n_ouput_= n_output
    n_ouput_= n_output//len(kernel_size)
    assert n_output%len(kernel_size)==0, "n_ouput must be divisible by len(kernel_size)"
    input= x
    input= se_layer(input, name_prefix= name_prefix + 'se_0')
    for i in range(len(kernel_size)):
        out1.append(cnn_block(input, n_ouput_, kernel_size[i], stride[i], padding[i], name_prefix= name_prefix + f'cnn_{i}_1_'))
    out1= Concatenate(name=name_prefix + 'concat')(out1)
    out1= Dropout(drop_out, name=name_prefix + 'dropout')(out1)
    out1= se_layer(out1, name_prefix= name_prefix + 'se_1')

    out2= []
    for i in range(len(kernel_size)):
        out2.append(cnn_block(out1, n_ouput_, kernel_size[i], stride[i], padding[i], name_prefix= name_prefix + f'cnn_{i}_2_'))
    out2= Concatenate(name=name_prefix + 'concat2')(out2)
    out2= Dropout(drop_out, name=name_prefix + 'dropout2')(out2)


    return Activation('relu', name=name_prefix + 'relu_')(x+ out2) # shape = (x+ n_output)
    # return Concatenate(name=name_prefix + 'concat3')([x, out2]) # shape = (x+ n_output)

def make_layer_multi(x, n_layer, n_output, dropRate=0.2, name_prefix=''):
    for i in range(n_layer):
        x= multi_branch_cnn(x, n_output, kernel_size=[3, 5, 7], stride=[1, 1, 1], padding=['same', 'same', 'same'], drop_out=dropRate, name_prefix= name_prefix + f'block{i+1}_')
    return x # shape = [n_layer*x+ (n_layer-1)*n_output]

def Multi_CNN_SE_DenseNet(input_shape, layers=[4, 4, 4, 4], outputs=[64*3]*4, head7x7= True, dropRate=0.2, num_classes=5):
    input= Input(shape=input_shape, name='input')
    headplanes= 64
    if head7x7:
        head= Conv2D(headplanes*3, kernel_size=7, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)
    else:
        head= Conv2D(headplanes, kernel_size=3, strides=2, padding='same', name='head_conv', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        head= BatchNormalization(name='head_bn')(head)
        head= Activation('relu', name='head_relu')(head)

        head= Conv2D(headplanes, kernel_size=3, strides=1, padding='same', name='head_conv2', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn2')(head)
        head= Activation('relu', name='head_relu2')(head)

        head= Conv2D(headplanes*3, kernel_size=3, strides=1, padding='same', name='head_conv3', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0001))(head)
        head= BatchNormalization(name='head_bn3')(head)
        head= Activation('relu', name='head_relu3')(head)
    head= MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='head_pool')(head)

    for (i, (n_layer, n_output)) in enumerate(zip(layers, outputs)):
        head= make_layer_multi(head, n_layer, n_output, dropRate=dropRate, name_prefix= f'block{i+1}_')
        # head= transition_block(head, headplanes, name_prefix= f'transition{i+1}_')

    head= BatchNormalization(name='head_bn4')(head)
    head= Activation('relu', name='head_relu4')(head)
    head= GlobalAveragePooling2D(name='head_gap')(head)
    head= Dense(num_classes, activation='softmax', name='head_fc')(head)
    model= Model(inputs=input, outputs=head)
    return model






