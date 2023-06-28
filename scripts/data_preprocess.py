# tạo train, test cho tubor
import os
import random
import shutil

def split_train_test_data(source_folder, train_folder, test_folder, test_ratio):
    # Tạo các thư mục train và test
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Lấy danh sách các thư mục con trong thư mục gốc
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    for subfolder in subfolders:
        class_name = os.path.basename(subfolder)
        train_class_folder = os.path.join(train_folder, class_name)
        test_class_folder = os.path.join(test_folder, class_name)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(test_class_folder, exist_ok=True)

        # Lấy danh sách các hình ảnh trong thư mục con
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]

        # Số lượng hình ảnh để chuyển sang tập kiểm tra
        num_test_images = int(len(images) * test_ratio)

        # Chọn ngẫu nhiên các hình ảnh để chuyển sang tập kiểm tra
        test_images = random.sample(images, num_test_images)

        # Di chuyển các hình ảnh sang thư mục train hoặc test tương ứng
        for image in images:
            if image in test_images:
                shutil.move(image, test_class_folder)
            else:
                shutil.move(image, train_class_folder)

# Thư mục chứa dữ liệu gốc
source_folder = '/media/data3/users/longnd/ML_prj/Data/TB_Chest_Radiography_Database'

# Thư mục train
train_folder = '/media/data3/users/longnd/ML_prj/Data/train'

# Thư mục test
test_folder = '/media/data3/users/longnd/ML_prj/Data/test'

# Tỉ lệ dữ liệu trong tập test
test_ratio = 0.2

# Tách dữ liệu thành train và test
split_train_test_data(source_folder, train_folder, test_folder, test_ratio)




def merge_folders(source_folder1, source_folder2, destination_folder):
    # Gộp hai thư mục
    shutil.move(source_folder1, destination_folder)
    shutil.move(source_folder2, destination_folder)

# Đường dẫn thư mục nguồn 1
source_folder1 = '/media/data3/users/longnd/ML_prj/Data/train/Normal'

# Đường dẫn thư mục nguồn 2
source_folder2 = '/media/data3/users/longnd/ML_prj/Data/train/NORMAL'

# Đường dẫn thư mục đích
destination_folder = '/media/data3/users/longnd/ML_prj/Data/train/NORMAL1'

# Gộp hai thư mục
merge_folders(source_folder1, source_folder2, destination_folder)

def move_subfolders_to_parent_folder(parent_folder):
    # Lấy danh sách các thư mục con trong thư mục lớn
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    images = [f.path for f in os.scandir(subfolders[0]) if f.is_file()]
    # Di chuyển các thư mục con ra ngoài thư mục lớn
    for subfolder in images:
        shutil.move(subfolder, parent_folder)

# Thư mục lớn
parent_folder = '/media/data3/users/longnd/ML_prj/Data/train/NORMAL1'

# Di chuyển các thư mục con ra ngoài thư mục lớn
move_subfolders_to_parent_folder(parent_folder)