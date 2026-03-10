import cv2 as cv
import numpy as np
import os
#khai bao doi tuong cap va face_recog
#lay username
username = input("Enter your username: ")
save_path = f"data/{username}"
if os.path.exists(save_path):
    print("Username already exists. Please choose a different username.")
    exit() 
else:
    os.makedirs(save_path)

cap = cv.VideoCapture(0)
face_recog = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")

dem = 0 #dem so luong anh da luu


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #chuyen doi den trang
    if frame is not None:
        bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    
    #khu nhieu
    if bw is not None:
        dbw = cv.GaussianBlur(bw, (5,5), 0)    
    
    #bounding box
    face = face_recog.detectMultiScale(dbw, 1.3, 5)
    #ve hinh chu nhat quanh khuon mat
    for (x,y,w,h) in face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        #luu anh vao thu muc
        face_img = dbw[y:y+h, x:x+w]
        cv.imwrite(f"{save_path}/{username}_{dem}.jpg", face_img)
        dem += 1
        
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q') or dem >= 100:
        break
cap.release()
cv.destroyAllWindows()