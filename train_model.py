import cv2 as cv
import numpy as np
import os

# Khai báo folder chứa data
data_path = "data"

#công cụ nhận diện khuôn mặt của ai
reg_tool = cv.face.LBPHFaceRecognizer_create() 

faces = []
labels = []
label_dict = {} #từ điển để map tên người với label số
current_label = 0

for user in os.listdir(data_path):
    user_path = os.path.join(data_path, user) 
    if not os.path.isdir(user_path):
        continue
    label_dict[current_label] = user #map label số với tên người
    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)
        img_bw = cv.imread(img_path, cv.IMREAD_GRAYSCALE) #đọc ảnh đen trắng
        faces.append(img_bw)
        labels.append(current_label)
    current_label += 1  

reg_tool.train(faces, np.array(labels))    #huấn luyện mô hình với dữ liệu đã thu thập
reg_tool.save("face_recognizer_model.yml") #lưu mô hình đã huấn luyện vào file yml
# lưu label_dict để có thể map lại tên người khi nhận diện
np.save("label_dict.npy", label_dict)
print("Model trained and saved successfully.")       