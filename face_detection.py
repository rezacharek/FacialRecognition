import cv2
import numpy as np


face = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt.xml')

if face.empty():
    raise IOError('Unable to read the file')


my_web_cam = cv2.VideoCapture(0)
scaling_factor = 1
c = cv2.waitKey(1)

# face_rects = face_cascade.detectMultiScale(img, 1.3, 5)
while c != 27:
    ret, new_frame = my_web_cam.read()
    new_frame = cv2.resize(new_frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)
    
    gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
    detected_assets_on_gray = face.detectMultiScale(gray_new_frame, 1.3, 5)
    for (x,y,w,h) in detected_assets_on_gray:
        cv2.rectangle(new_frame, (x,y), (x+w,y+h), (0,255,0), 3)

    cv2.imshow('Resized',new_frame)
    c = cv2.waitKey(1)