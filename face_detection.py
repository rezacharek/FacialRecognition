import cv2
import numpy as np


face = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt.xml')
if face.empty():
    raise IOError('Unable to read the file of face')

eye = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
if eye.empty():
    raise IOError('Unable to read the file of eye')


my_web_cam = cv2.VideoCapture(0)
scaling_factor = 0.2
c = cv2.waitKey(1)
while c != 27:
    ret, new_frame = my_web_cam.read()
    new_frame = cv2.resize(new_frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)
    
    gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
    detected_assets_on_gray_face = face.detectMultiScale(gray_new_frame, 1.3, 5)
    detected_assets_on_gray_eye = eye.detectMultiScale(gray_new_frame, 1.3, 5)
    

    
    for (x,y,w,h) in detected_assets_on_gray_face:
        cv2.rectangle(new_frame, (x,y), (x+w,y+h), (0,255,0), 3)

    
    for (x_eye, y_eye, w_eye, h_eye) in detected_assets_on_gray_eye:
        center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
        radius = int(0.3 * (w_eye + h_eye))
        color = (0, 255, 0)
        thickness = 3
        cv2.circle(new_frame, center, radius, color, thickness)
    

    cv2.imshow('Resized',new_frame)
    c = cv2.waitKey(1)