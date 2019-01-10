import socket
import time
from freenect2 import Device, FrameType
import cv2
import errno

HOST = '0.0.0.0'
PORT = 50505
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

sock.bind((HOST, PORT))
print('Socket bind complete')

sock.listen(5)
print('Socket now listening')

device = Device()

def signal_handler(signal=None, frame=None):
    exit(0)

def capture(server_socket):
    with device.running():
        for type_, frame in device:
            if type_ is FrameType.Color:
                try:
                    data = cv2.imencode('.jpg', frame.to_array())[1].tostring()
                    send(server_socket,data)
                except KeyboardInterrupt:
                    signal_handler()
                except socket.error:
                    return

def send(c, data):
    try:
        c.send(data)
        c.send(b"END!") # send param to end loop in client
    except socket.error as e:
        if e.errno == errno.EPIPE:
            raise e
        print('Error: {}'.format(e))

while True:
    server_socket, addr = sock.accept()
    capture(server_socket)
