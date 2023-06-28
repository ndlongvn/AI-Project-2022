from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization, GRU, Bidirectional
def CNN_Model(input_shape= (128, 128, 1), num_classes= 5, block1=True, block2=True, block3=True, block4=True, block5=True, regularizer=keras.regularizers.l2(0.0001), Dropout_ratio=0.15):

    model = keras.Sequential()

    model.add(keras.Input(shape=input_shape))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=block1, kernel_regularizer=regularizer, name= 'conv1_1'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=block1, kernel_regularizer=regularizer, name= 'conv1_2' ))
    model.add(BatchNormalization(name= 'bn1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name= 'maxpool1'))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', trainable=block2, kernel_regularizer=regularizer, name= 'conv2_1'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', trainable=block2, kernel_regularizer=regularizer, name= 'conv2_2'))
    model.add(BatchNormalization(name= 'bn2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name= 'maxpool2'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', trainable=block3, kernel_regularizer=regularizer, name= 'conv3_1'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', trainable=block3, kernel_regularizer=regularizer, name= 'conv3_2'))
    model.add(BatchNormalization(name= 'bn3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name= 'maxpool3'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block4, kernel_regularizer=regularizer, name= 'conv4_1'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block4, kernel_regularizer=regularizer, name= 'conv4_2'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block4, kernel_regularizer=regularizer, name= 'conv4_3'))
    model.add(BatchNormalization(name= 'bn4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name= 'maxpool4'))


    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block5, kernel_regularizer=regularizer, name= 'conv5_1'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block5, kernel_regularizer=regularizer, name= 'conv5_2'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', trainable=block5, kernel_regularizer=regularizer, name= 'conv5_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name= 'maxpool5'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', name= 'dense1'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization(name= 'bn5'))

    model.add(Dense(num_classes, activation='softmax', name= 'dense2'))
    return model

