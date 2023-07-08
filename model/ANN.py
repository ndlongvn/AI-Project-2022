from keras.layers import Dense, Flatten, Dropout, Input
from keras.models import Model
def MLP_Model(input_shape= (128, 128, 1), num_classes= 5,  dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    flatten_layer = Flatten()(inputs)
    dense_layer = Dense(512, activation='relu', name= 'dense_1')(flatten_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(256, activation='relu', name= 'dense_2')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(128, activation='relu', name= 'dense_3')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(64, activation='relu', name= 'dense_4')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(32, activation='relu', name= 'dense_5')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(16, activation='relu', name= 'dense_6')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    outputs = Dense(num_classes, activation='softmax', name= 'output')(dense_layer)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def ANN_Model(input_shape= (128, 128, 3), num_classes= 5, dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    flatten_layer = Flatten()(inputs)
    dense_layer = Dense(64, activation='relu', name= 'dense_1')(flatten_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(128, activation='relu', name= 'dense_2')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(256, activation='relu', name= 'dense_3')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(128, activation='relu', name= 'dense_4')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    dense_layer = Dense(32, activation='relu', name= 'dense_5')(dense_layer)
    dense_layer= Dropout(dropout_rate)(dense_layer)
    outputs = Dense(num_classes, activation='softmax', name= 'output')(dense_layer)
    model = Model(inputs=inputs, outputs=outputs)
    return model