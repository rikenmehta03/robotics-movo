import cv2
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kinect import ObjectDetector

od = ObjectDetector()
counter = 0
start_time = time.time()
for frame_data in od.extract():
    if (time.time() - start_time) > 1:                
        print("FPS: {}".format(counter))
        counter = 0
        start_time = time.time()
    frame = frame_data['image']
    try:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('video', frame)
        counter+=1
        if cv2.waitKey(10) == ord('q'):
            break
    except Exception as e:
        print('{}'.format(e))
        break
