import streamlit as st
import pandas as pd
import cv2

# Tạo DataFrame mẫu
data = {
    'Tên': ['Hình ảnh 1', 'Hình ảnh 2', 'Hình ảnh 3'],
    'Đường dẫn': ['image/coffee.jpg', 'image/coffee.jpg', 'image/coffee.jpg']
}
df = pd.DataFrame(data)
st.dataframe(df['Tên'])

# Kiểm tra xem tên hình ảnh nào được nhấp
clicked_name = st.selectbox('Chọn tên hình ảnh', df['Tên'])

# Tìm đường dẫn hình ảnh tương ứng với tên đã chọn
image_path = df.loc[df['Tên'] == clicked_name, 'Đường dẫn'].values[0]

# Hiển thị hình ảnh
if image_path:
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    st.image(image, caption=clicked_name, use_column_width=True)

import pandas as pd
import streamlit as st

# Tạo DataFrame
data = {'Name': ['John', 'Jane', 'Mike'],
        'Age': [25, 30, 35],
        'Button': ['<button>Click</button>', '<button>Click</button>', '<button>Click</button>']}
df = pd.DataFrame(data)

# Tạo hàm để chuyển đổi cột 'Button' sang HTML
def button_formatter(button):
    image = cv2.imread('image/coffee.jpg', cv2.COLOR_BGR2RGB)
    st.image(image, caption=clicked_name, use_column_width=True)

# Áp dụng hàm button_formatter cho cột 'Button'
df['Button'] = df['Button'].apply(button_formatter)

# Hiển thị DataFrame với ô dữ liệu chứa nút
st.dataframe(df, unsafe_allow_html=True)

    # st.markdown("""            
    #         **MY TEAM**:
    #         - <a href='https://www.facebook.com/'>Nguyen Duy Long</a>
    #         - <a href='https://www.facebook.com/'>Ho Viet Duc Luong</a>
    #         - <a href='https://www.facebook.com/'>Ngo Tran Anh Thu</a>
    #         - <a href='https://www.facebook.com/'>Pham Xuan Truong</a>

    #         **APP INFO**:
    #         - This app is built using Streamlit, a Python library for building web apps.
    #         - The model is trained using Convolutional Neural Network (CNN) and Artificial Neural Network (ANN).
    #         - The model is trained using 4,000 images of X-Ray images.

    #         **DATA INFO**:
    #         - The data is collected from Kaggle.
    #         - The data is collected from 4 different sources:
    #             + https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    #             + https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    #             + https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities
    #             + https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    #     """, unsafe_allow_html=True)
