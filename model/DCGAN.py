import tensorflow as tf
from keras import layers

# Hàm tạo generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,), name= 'dense_1'))
    model.add(layers.BatchNormalization(name='bn_1'))
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) 

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name= 'conv2d_transpose_1'))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization(name='bn_2'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, name= 'conv2d_transpose_2'))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization(name='bn_3'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', name= 'conv2d_transpose_3'))
    assert model.output_shape == (None, 64, 64, 32)

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', name= 'conv2d_transpose_4'))
    assert model.output_shape == (None, 128, 128, 3)

    return model

# Hàm tạo discriminator
def make_discriminator_model(input_shape= [128, 128, 3]):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape, name= 'conv2d_1'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name= 'conv2d_2'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', name= 'conv2d_3'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, name= 'dense_1'))

    return model
