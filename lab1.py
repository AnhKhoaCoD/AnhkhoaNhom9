import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

base_frame = None
while True:
    ret, frame = cam.read()
    if not ret:
        break

    if frame is not None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)
    if base_frame is None:
        base_frame = gray
        continue
    chenh_lech = cv.absdiff(base_frame, gray)
    nguong = cv.threshold(chenh_lech, 25, 255, cv.THRESH_BINARY)[1]
    nguong = cv.dilate(nguong, None, iterations=2)
    bien, info = cv.findContours(nguong.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for b in bien:
        if cv.contourArea(b) <= 700:
            continue    
        (x, y, w, h) = cv.boundingRect(b)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("Camera", frame)
        cv.imshow("Threshold", nguong)


    if cv.waitKey(10) == ord('q'):
        break

cv.destroyAllWindows()