from keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2, DenseNet121
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
def VGG16_Model(input_shape= (128, 128, 1), num_classes= 5):
    model = Sequential()
    model.add(VGG16(weights='imagenet', include_top=False, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def ResNet50_Model(input_shape= (128, 128, 1), num_classes= 5):
    model = Sequential()
    model.add(ResNet50(weights='imagenet', include_top=False, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def InceptionV3_Model(input_shape= (128, 128, 1), num_classes= 5):
    model = Sequential()
    model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def MobileNetV2_Model(input_shape= (128, 128, 1), num_classes= 5):
    model = Sequential()
    model.add(MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def DenseNet121_Model(input_shape= (128, 128, 1), num_classes= 5):
    model = Sequential()
    model.add(DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
