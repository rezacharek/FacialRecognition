import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)
i = 0
scaling_factor = 0.3
c = cv2.waitKey(1)
while c != 27:
    # raw_input('Press Enter to capture')
    time.sleep(0.1)
    return_value, image = camera.read()
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)
    gray_new_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Showing',image)
    c = cv2.waitKey(1)

    cv2.imwrite('image_'+str(i)+'.png', image)
    i += 1
del(camera)