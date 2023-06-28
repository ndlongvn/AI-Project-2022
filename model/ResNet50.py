from keras import layers
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Concatenate, Multiply, AveragePooling2D, MaxPooling2D, Reshape, Input

def convolutional_block(x, filters, strides=1, name_prefix=''):
    shortcut = x
    if strides != 1 or x.shape[-1] != 4 * filters:
        shortcut = tf.keras.layers.Conv2D(4 * filters, (1, 1), strides=strides, name=name_prefix + 'shortcut_conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=name_prefix + 'shortcut_bn')(shortcut)
    
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, name=name_prefix + 'conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn1')(x)
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu1')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=name_prefix + 'conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn2')(x)
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu2')(x)
    
    x = tf.keras.layers.Conv2D(4 * filters, (1, 1), name=name_prefix + 'conv3')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn3')(x)
    
    x = tf.keras.layers.Add(name=name_prefix + 'add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu3')(x)
    return x

def identity_block(x, filters, name_prefix=''):
    shortcut = x
    
    x = tf.keras.layers.Conv2D(filters, (1, 1), name=name_prefix + 'conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn1')(x)
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu1')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=name_prefix + 'conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn2')(x)
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu2')(x)
    
    x = tf.keras.layers.Conv2D(4 * filters, (1, 1), name=name_prefix + 'conv3')(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + 'bn3')(x)
    
    x = tf.keras.layers.Add(name=name_prefix + 'add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name_prefix + 'relu3')(x)
    return x

def ResNet50_Model_No_Weight(input_shape=(128, 128, 1), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)
    
    x = convolutional_block(x, 64, strides=1, name_prefix='conv2_')
    x = identity_block(x, 64, name_prefix='id2a_')
    x = identity_block(x, 64, name_prefix='id2b_')
    
    x = convolutional_block(x, 128, strides=2, name_prefix='conv3_')
    x = identity_block(x, 128, name_prefix='id3a_')
    x = identity_block(x, 128, name_prefix='id3b_')
    x = identity_block(x, 128, name_prefix='id3c_')
    
    x = convolutional_block(x, 256, strides=2, name_prefix='conv4_')
    x = identity_block(x, 256, name_prefix='id4a_')
    x = identity_block(x, 256, name_prefix='id4b_')
    x = identity_block(x, 256, name_prefix='id4c_')
    x = identity_block(x, 256, name_prefix='id4d_')
    x = identity_block(x, 256, name_prefix='id4e_')
    
    x = convolutional_block(x, 512, strides=2, name_prefix='conv5_')
    x = identity_block(x, 512, name_prefix='id5a_')
    x = identity_block(x, 512, name_prefix='id5b_')
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs, x)
    return model


