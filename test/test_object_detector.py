import cv2
from kinect import ObjectDetector
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


od = ObjectDetector()

for frame_data in od.extract():
    frame = frame_data['image']
    try:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == ord('q'):
            break
    except Exception as e:
        print('{}'.format(e))
        break
