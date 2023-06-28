# AI-Project-2022

### How to prepare dataset

1. Download dataset from Kaggle link below:
 - [Covid19 Dataset](http://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
 - [Tuberculosis Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
 - [Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
 - [Pneumothorax Dataset](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks)
2. Using code from <a href="https://github.com/ndlongvn/AI-Project-2022/blob/main/scripts/data_preprocess.py">data preprocess</a> to prepare dataset: 
### Model

<details open>

<summary>Table 1. Model Checkpoints.</summary>

<p> </p>

|           Architecture        |   Acc (%) |                                            Url                                           | Size (MB)|
|:-----------------------------:|:--------:|:----------------------------------------------------------------------------------------:|:--------:|
|ANN |   56  |[GoogleDrive](https://drive.google.com/file/d/1gVAakBi_hr5Q5IMKHJZwEE_UjjwojVgY/view?usp=drive_link) |    97  |
|CNN        |   87  |[GoogleDrive](https://drive.google.com/file/d/1U1qMT7jLaRvfAL7QPqt6xJ-BccoqqWxC/view?usp=drive_link) |    56  |
|InceptionV3       |   88  |[GoogleDrive](https://drive.google.com/file/d/1MKnFLCrOY251ClDGsHI9WiYUpeLZFo6D/view?usp=drive_link) |    85  |
|ResNet50        |   90   |[GoogleDrive](https://drive.google.com/file/d/13IDUZwuPN3msVN-Rx3BsBQ23UlACsuO6/view?usp=drive_link) |    91  |
|DenseNet121|   91   |[GoogleDrive](https://drive.google.com/file/d/12PTu5_CjryaFT0RaHZqIipjnwthv_eXJ/view?usp=drive_link) |    28   |

</details>


### How to using this app
1. Access to cloned folder from GitHub and install all requirements:
```Shell
pip install -r requirements.txt
```
2. Run app:
```Shell
streamlit run src/main.py
```
