import cv2 as cv
import numpy as np
import os


data_path = "data"

reg_tool = cv.face.LBPHFaceRecognizer_create()

faces = []
labels = []
labels_dict = {}
current_label = 0

for user in os.listdir(data_path):
    user_path = os.path.join(data_path, user)
    if not os.path.isdir(user_path):
        continue
    labels_dict[current_label] = user
    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)
        img_bw = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        faces.append(img_bw)
        labels.append(current_label)
    current_label += 1

reg_tool.train(faces, np.array(labels))

reg_tool.save("face_recog_model.yml")
np.save("labels_dict.npy", labels_dict)
print("Model trained and saved successfully.")