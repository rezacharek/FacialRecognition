import cv2
import numpy as numpy

# Opening the web Cam
my_web_cam = cv2.VideoCapture(0)


c = cv2.waitKey(1)
scaling_factor = 1

# Capturing new Frames
while(c != 27):
    ret, new_frame = my_web_cam.read()
    new_frame = cv2.resize(new_frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized', new_frame)
    c = cv2.waitKey(1)

cv2.destoryAllWindows()