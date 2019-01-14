import socket
import cv2
import numpy as np
import time
import sys
import errno
from contextlib import contextmanager

class CameraStream:
    def __init__(self, ip, port, packet_size=8192, verbose=True):
        self.ip = ip
        self.port = port
        self.packet_size = packet_size
        self.is_open = False
        self.verbose = verbose
    
    def __iter__(self):
        def iterator():
            start_time = time.time()
            counter = 0
            while True:
                try:
                    frame = self.get_next_frame()
                    if (time.time() - start_time) > 1:
                        if self.verbose:
                            print("FPS: {}".format(counter))
                        counter = 0
                        start_time = time.time()
                    if frame is not None:
                        counter+=1
                        yield frame
                except socket.error as e:
                    raise e
        return iterator()
    
    def open(self):
        print("Opening video stream")
        self.stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.stream.connect((self.ip, self.port))
            self.is_open = True
        except socket.error as err:
            print("Connection error. Possible issues:")
            print("- Wrong ip")
            print("- Wrong port")
            print("kinnect_server.py not running on robot PC")
            raise err

    def close(self):
        print('closing video stream')
        self.stream.close()
        self.is_open = False

    def get_next_frame(self):
        if self.is_open:
            data = b''
            while True:
                try:
                    r = self.stream.recv(self.packet_size)
                    if len(r) == 0:
                        continue
                    a = r.find(b'END!')
                    if a != -1:
                        data += r[:a]
                        break
                    data += r
                except socket.error as e:
                    raise e
                except Exception:
                    pass

            nparr = np.fromstring(data, np.uint8)
            
            if nparr.shape[0] == 0:
                return None
            else:
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                return frame
        else:
            raise Exception('Stream not open. use `open()` before fetching next frame. Or use `with .running()`')
        
    @contextmanager
    def running(self):
        self.open()
        yield self
        self.close()
        