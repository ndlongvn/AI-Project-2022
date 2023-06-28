import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Metric
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Định nghĩa một subclass của Metric trong Keras
class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=(-1,))
        y_pred = tf.reshape(y_pred, shape=(-1,))
        true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
        false_positives = tf.reduce_sum(tf.cast((1 - y_true) * tf.round(y_pred), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(y_true * (1 - tf.round(y_pred)), tf.float32))
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
# Compile model
def model_compiling(model, loss = tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001)):
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', 'Precision', 'Recall', F1Score()])

# Load data
def get_data(directory, size= (128, 128, 1), batch_size= 64, val_size= 0.1, do_val= False):
    # size of image
    img_hight, img_width = size[0], size[1]
    Train_data = pd.DataFrame(columns=['path', 'class'])
    Test_data = pd.DataFrame(columns=['path', 'class'])

    for filename in ['train', 'test']:
        for filename2 in os.listdir(directory+'/'+filename):
            for images in os.listdir(directory+'/'+filename+'/'+filename2):
                if filename =='train':
                    Train_data = pd.concat([Train_data, pd.DataFrame({'path': directory+'/'+filename+'/'+filename2+'/'+images , 'class': filename2.upper()}, index=[0])], ignore_index=True)
                if filename =='test':
                    Test_data = pd.concat([Test_data, pd.DataFrame({'path': directory+'/'+filename+'/'+filename2+'/'+images , 'class': filename2.upper()}, index=[0])], ignore_index=True)
    # split to val and train data
    
    classes = get_classes()
    
    train_data_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.1, width_shift_range=0.05, height_shift_range=0.05
                                        , horizontal_flip=False, fill_mode='nearest', rotation_range=5, shear_range=0.05,
                                        brightness_range=[0.9, 1.1]) 
    test_data_gen = ImageDataGenerator(rescale=1./255) 
    if do_val:
        train, val = train_test_split(Train_data, test_size=val_size, random_state=42, stratify=Train_data['class'])
        train_data = train_data_gen.flow_from_dataframe(train, x_col='path', y_col='class',
                                                image_size=(img_hight, img_width), target_size=(
                                                    img_hight, img_hight), color_mode='rgb',
                                                batch_size=batch_size, class_mode='categorical',
                                                classes=classes, subset=None, shuffle= True)

        val_data = test_data_gen.flow_from_dataframe(val, x_col='path', y_col='class',
                                                    image_size=(img_hight, img_width), target_size=(
                                                        img_hight, img_hight), color_mode='rgb',
                                                    batch_size=batch_size, class_mode='categorical',
                                                    classes=classes, subset=None)
        test_data = test_data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                                    image_size=(img_hight, img_width), target_size=(
                                                        img_hight, img_hight), color_mode='rgb',
                                                    batch_size=batch_size,class_mode='categorical',
                                                    classes=classes, subset=None)
        
        return train_data, val_data, test_data
    else:
        train_data = train_data_gen.flow_from_dataframe(Train_data, x_col='path', y_col='class',
                                                image_size=(img_hight, img_width), target_size=(
                                                    img_hight, img_hight), color_mode='rgb',
                                                batch_size=batch_size, class_mode='categorical',
                                                classes=classes, subset=None, shuffle= True)
        test_data = test_data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                                    image_size=(img_hight, img_width), target_size=(
                                                        img_hight, img_hight), color_mode='rgb',
                                                    batch_size=batch_size,class_mode='categorical',
                                                    classes=classes, subset=None, shuffle= False)
        return train_data, test_data

# Get class weight
def get_balanced_weight(train_data):
    df= pd.DataFrame([])
    df['class']= train_data.labels
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(df['class']), y= df['class'])
    class_weights = dict(enumerate(class_weights))
    return class_weights

# Plot training curve
def plot_history(history):
    # Lấy các giá trị loss và accuracy từ lịch sử
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    
    # Lấy các giá trị validation loss và validation accuracy nếu có
    val_loss = history.history.get('val_loss')
    val_accuracy = history.history.get('val_accuracy')
    
    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Vẽ biểu đồ accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy, label='Training Accuracy')
    if val_accuracy:
        plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

 
# Get report
def get_report(model, test_data):
    y_pred = model.predict(test_data, workers=4, use_multiprocessing=True)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = test_data.labels
    target_names = list(test_data.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

# Get classes
def get_classes():
    classes = ['COVID19', 'NORMAL1', 'PNEUMONIA','TUBERCULOSIS', 'PNEUMOTHORAX']
    return classes

# Save history
def save_history_to_csv(history, filename):
    # Tạo DataFrame từ dictionary lịch sử
    history_df = pd.DataFrame(history.history)
    
    # Lưu DataFrame thành tệp CSV
    history_df.to_csv(filename, index=False)

