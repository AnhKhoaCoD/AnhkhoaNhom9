import cv2 as cv 
import numpy as np 
import os
import webbrowser   # thêm dòng này

#khai báo folder chứa data
#lấy user name:
user_name = input("Enter your name:")

# nếu tên là Khai thì mở link
if user_name.lower() == "khai":
    webbrowser.open("https://www.google.com")   # đổi link tùy bạn

save_path = f"data/{user_name}"

if os.path.exists(save_path):
    print("User already exists. Please choose a different name.")
    exit()
else:
    os.makedirs(save_path)

cap = cv.VideoCapture(0)
face_reg = cv.CascadeClassifier(cv.data.haarcascades +"haarcascade_frontalface_default.xml")

dem = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if frame is not None:
        bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dbw = cv.GaussianBlur(bw, (5, 5), 0)

        face = face_reg.detectMultiScale(dbw, 1.3, 5)

        for (x,y,w,h) in face:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

            face_img = dbw[y:y+h, x:x+w]
            cv.imwrite(f"{save_path}/{dem}.jpg", face_img)
            dem += 1

    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q') or dem >= 100:
        break

cap.release()
cv.destroyAllWindows()