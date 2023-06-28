import cv2
import os
import numpy as np

# define preprocessing image
def preprocess_image_gray(image_path):
    img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img_arr, (128, 128))
    img = img/255
    img= np.array(img).reshape(128, 128, 1)
    return img

def preprocess_image_rgb(image_path):
    img_arr = cv2.imread(image_path)
    img = cv2.resize(img_arr, (128, 128))
    img = img/255
    return img

def preprocess_image(list_image_path):
    img= []
    for image_path in list_image_path:
        img.append(preprocess_image_rgb(image_path))
    return np.array(img)

# load model we need
def load_model(model, model_path):
    model.load_weights(model_path)
    if model.weights:
        print('Weights loaded successfully.')
    else:
        print('Error loading weights.')
    return model

def softmax(x):
  e_x = np.exp(x)
  sum_exp_x = np.sum(e_x, axis=-1, keepdims=True)
  return e_x / sum_exp_x

# define prediction
def predict(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction*100

# make url clickable   
def make_clickable(link):
    # display for your link
    text = '🔗' # link.split('/')[-1]
    # fb= 'https://www.facebook.com/'
    fb= link
    img_path=  f'<a target="_blank" href="{fb}">{text}</a>' # link
    return img_path

# save upload file if needed
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static file/image', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0  
    
# highlight high and low value
def color_value(val):
    value= float(val[:-1])
    if value <= 20:
        return 'color: green'
    elif value > 20 and value < 80:
        return 'color: orange'
    else:
        return 'color: red'
    
def get_name(name):
    name= name.split('.')[0]
    if len(name)<12:
        return name
    else:
        return name[:12]+'...'
    
def inverse_softmax(row):
    logit = np.log(row)
    inv_softmax = np.exp(logit) / np.sum(np.exp(logit))
    return inv_softmax