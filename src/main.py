from function import *
from page import page_group
import streamlit as st
import seaborn as sns
import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# parent_dir = os.path.dirname(os.path.__file__)
import tensorflow as tf
sys.path.append(parent_dir)
from model.ANN import MLP_Model
from model.CNN import CNN_Model
from model.Finetune import ResNet50_Model, DenseNet121_Model
import time
import cv2
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

input_shape= (128, 128, 3)
num_classes= 5
    

# run main
if __name__ == '__main__':

    # set up the page
    sns.set_theme(style= 'darkgrid')
    sns.set()
    st.set_page_config(
        page_title="AI APP- Fun with AI 🐧️",
        page_icon="🧊",
        initial_sidebar_state="auto",
        layout='wide')
    # load model

    with st.spinner('Loading model...'):
        # dict_model= {'ANN': MLP_Model(input_shape, num_classes),
        #             'CNN': CNN_Model(input_shape, num_classes),
        #             'ResNet50': ResNet50_Model(input_shape, num_classes),
        #             'DenseNet121 (Best Model)': DenseNet121_Model(input_shape, num_classes)}
        dict_model_path= {'ANN': 'checkpoints/ann_weights.h5', 
                        'CNN': 'checkpoints/cnn.h5', 
                        'ResNet50': 'checkpoints/resnet50.h5', 
                        'DenseNet121 (Best Model)': 'checkpoints/densenet121.h5'}
        time.sleep(1)
    
    lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
    st_lottie(lottie_book, speed=1, height=200, key="initial")

    # st.title('Lung Disease Detector')
    t1, t2 = st.columns((5, 4))
    with t1:
        st.markdown('# 🫁 Lung Disease Detector')

    with t2:
        st.write("")
        st.write("")
        st.write("""
        **Computer Science (_Talented Program_)** | Ha Noi University of Science and Technology
        """)

    instructions = """
            * This is a simple web app to classify some popular disease from X-Ray images. 
            * The dataset is from Kaggle.
            * The model using ANN, CNN, ResNet50, DenseNet121 (Best Model).
            """
    st.markdown(instructions)

    # slide bar for select disease and model
    st.sidebar.markdown("## 🎈 Select Disease and Model")

    with st.sidebar:
        with st.expander("✨ Model", True):
            st.checkbox('ANN', key='ANN_model')
            st.checkbox('CNN', key='CNN_model')
            st.checkbox('ResNet50', key='ResNet50_model')
            st.checkbox('DenseNet121 (Best Model)', key='DenseNet121_model', value= True)
        with st.expander("❄️ Disease", True):
            st.checkbox('Covid19', key='Covid19_di')
            st.checkbox('Pneumonia', key='Pneumonia_di')
            st.checkbox('Tuberculosis', key='Tuberculosis_di')
            st.checkbox('Pneumothorax', key='Pneumothorax_di')
            st.checkbox('All', key='All_di', value= True)
        st.sidebar.button(label='Clear Cache')
    
    # check model
    use_model= 'ResNet50'
    use_disease= ['COVID19', 'PNEUMONIA', 'TUBERCULOSIS', 'PNEUMOTHORAX']
    if st.session_state.ANN_model:
        use_model = 'ANN'
    elif st.session_state.CNN_model:
        use_model = 'CNN'
    elif st.session_state.ResNet50_model:
        use_model = 'ResNet50'
    elif st.session_state['DenseNet121_model']:
        use_model = 'DenseNet121 (Best Model)'

    # check disease    

    if 'All_di' not in st.session_state:
        st.session_state.All_di = True

    use_disease= []
    if st.session_state.Covid19_di:
        use_disease.append('COVID19')
    if st.session_state.Pneumonia_di:
        use_disease.append('PNEUMONIA')
    if st.session_state.Tuberculosis_di:
        use_disease.append('TUBERCULOSIS')
    if st.session_state.Pneumothorax_di:
        use_disease.append('PNEUMOTHORAX')

    if st.session_state.All_di:
        use_disease = ['COVID19', 'PNEUMONIA', 'TUBERCULOSIS', 'PNEUMOTHORAX']

    
       
    
    
    # upload file button accept for many file type
    st.markdown('### Please Upload Images')
    allowed_types = ['jpg', 'jpeg', 'png']
    # @st.cache(allow_output_mutation=True)
    uploaded_files = st.file_uploader("Upload yours Images here", type= allowed_types, accept_multiple_files= True)

    
    st.subheader(f"Predict For Image")     
    # ['COVID19', 'NORMAL', 'PNEUMONIA','TUBERCULOSIS', 'PNEUMOTHORAX']
    normal= [] # normal
    tb= [] # TB
    pneu= [] # Pneumonia
    cov= [] # Covid19
    pne= [] # Pneumothorax
    
    img_path= [os.path.join(parent_dir, 'image', uploaded_file.name).replace('\\', '/') for uploaded_file in uploaded_files]
    # img_path= [os.path.join(parent_dir, 'image', uploaded_file.name).replace('/', '\\') for uploaded_file in uploaded_files]
    if uploaded_files is not None and len(uploaded_files) != 0:   
            if st.button('Predict For Image'): 
                # load model and prediction
                with st.spinner('Wait for prediction...'):
                    if use_model == 'ANN':
                        model= load_model(MLP_Model(input_shape= input_shape, num_classes= num_classes), os.path.join(parent_dir, dict_model_path[use_model]))#.replace('/', '\\'))
                    elif use_model == 'CNN':
                        model= load_model(CNN_Model(input_shape= input_shape, num_classes= num_classes), os.path.join(parent_dir, dict_model_path[use_model]))#.replace('/', '\\'))
                    elif use_model == 'ResNet50':
                        model= load_model(ResNet50_Model(input_shape= input_shape, num_classes= num_classes), os.path.join(parent_dir, dict_model_path[use_model]))#.replace('/', '\\'))
                    elif use_model == 'DenseNet121 (Best Model)':
                        model= load_model(DenseNet121_Model(input_shape= input_shape, num_classes= num_classes), os.path.join(parent_dir, dict_model_path[use_model]))#.replace('/', '\\'))
                    preds= predict(img_path, model)
                    for i, pred in enumerate(preds):
                        normal.append(pred[1])
                        tb.append(pred[3])
                        pneu.append(pred[2])
                        cov.append(pred[0])
                        pne.append(pred[4])
                    
                    # data frame for prediction
                    df= pd.DataFrame({'Image'.upper() :[str(i) for i in range(len(uploaded_files))], 
                                    'Name'.upper(): [get_name(uploaded_files[i].name) for i in range(len(uploaded_files))], 
                                        'Normal'.upper(): normal, 'Covid19'.upper(): cov, 
                                        'Pneumonia'.upper(): pneu,'Tuberculosis'.upper(): tb, 
                                        'Pneumothorax'.upper(): pne, 
                                        'Link'.upper(): img_path})         
                    # show data frame
                    df_new= df[['NORMAL']+ use_disease].copy() 
                    # df_new= df_new.apply(inverse_softmax, axis=1)
                    if df.shape[0] != 0:
                        data = df_new.copy().to_numpy()
                        softmax_data = tf.nn.softmax(data, axis=1).numpy()
                        softmax_data = np.round(softmax_data, 2) * 100
                        df_softmax = pd.DataFrame(softmax_data, columns=df_new.columns)
                        df_softmax['Image'.upper()] = df['Image'.upper()]
                        df_softmax['Link'.upper()] = df['Link'.upper()]
                        df_softmax['Name'.upper()] = df['Name'.upper()]

                        df= df_softmax.copy()

                    # 
                    for i in ['NORMAL']+ use_disease:
                        if i != 'Link'.upper() and i != 'Image'.upper() and i != 'Name'.upper():
                            df[i] = df[i].apply(lambda x: '{:.2f}%'.format(x)).apply(lambda x: f'<span style="{color_value(x)}">{x}</span>').apply(lambda x: f'<span style="font-size: 18px">{x}</span>')
                    df= df[['Image'.upper(), 'Link'.upper(), 'Name'.upper()]+ ['NORMAL']+ use_disease] 
                    
                    df1= df[['Name'.upper()]+ ['NORMAL']+ use_disease].copy()

                # show data frame    
                html = f'<div style="width: 450px; font-size: 14px; text-align:center ">{df1.to_html(index=False, escape=False)}</div>'
                st.write(html, unsafe_allow_html=True)
                st.write("")
                # show image
                with st.expander("See more about image"):
                    img_list= []
                    img_name= []
                    img_cotent= []
                    for i in df.index:
                        img_list.append(preprocess_image_rgb(df['Link'.upper()][i]))
                        img_name.append(df['Name'.upper()][i])
                        predd= []
                        for j in ['NORMAL']+ use_disease:
                            predd.append((j, df[j][i]))
                        img_cotent.append(predd)
                    # show image
                    for i in range(len(img_list)):
                        c1, c2= st.columns((4, 3))
                        with c1:
                            st.image(img_list[i], width= 300, caption= img_name[i])
                        with c2:
                            st.markdown("<hr>", unsafe_allow_html=True)
                            for m, n in img_cotent[i]:
                                st.markdown(f"<div style='width: 300px; display: inline-block;'>{m}: {n}</div>", unsafe_allow_html=True)


                time.sleep(0.5)

    # more app info
    st.markdown("### About This Web App")
    st.markdown("""
                <p2> This Web App is developed by AntiLungDisease Team: \n
                   Nguyen Duy Long\n
                   Ho Viet Duc Luong\n
                   Ngo Tran Anh Thu\n
                   Pham Xuan Truong\n
                If you want to support us with money, just click  </p2>
                <a href='https://drive.google.com/file/d/1y4SURr2yZQhuKIQwka8sJRr7um0Xx6lo/view?usp=sharing'> here:</a>
                """, unsafe_allow_html=True)
    # st.image('image/coffee.jpg', width= 200)






