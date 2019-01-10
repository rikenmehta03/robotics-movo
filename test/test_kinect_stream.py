from kinect import CameraStream
import cv2

stream = CameraStream('10.66.171.1', 50505)

with stream.running():
    for frame in stream:
        try:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('video',frame)
            if cv2.waitKey(10) == ord('q'):
                break
        except Exception as e:
            print('{}'.format(e))
            break

