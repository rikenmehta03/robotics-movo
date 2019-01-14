import sys
import base64
import uuid
import json
import numpy as np
import redis
import queue
import threading

from .camera import CameraStream


class ObjectDetector:
    def __init__(self):
        self._redis_db = redis.StrictRedis(
            host='localhost', port='6379', db='0')
        self.stream = CameraStream('10.66.171.1', 50505)
        self.image_queue = queue.Queue()

    def _base64_encode(self, a):
        a = a.copy(order='C')
        return base64.b64encode(a).decode("utf-8")

    def _base64_decode(self, a, shape=None):
        if sys.version_info.major == 3:
            a = bytes(a, encoding="utf-8")

        a = np.frombuffer(base64.decodestring(a), dtype=np.float32)
        if shape is not None:
            a = a.reshape(shape)
        return a

    def _decode_redis_data(self, data):
        data = json.loads(data.decode('utf-8'))
        data['image'] = self._base64_decode(data['image'])
        return data

    def _publish_stream(self, stop_event):
        with self.stream.running():
            for frame in self.stream:
                if stop_event.is_set():
                    break
                frame = frame.astype(np.float32)
                _id = str(uuid.uuid4())
                data = {
                    'id': _id,
                    'image': self._base64_encode(frame)
                }
                self.image_queue.put(_id)
                self._redis_db.rpush('image_queue', json.dumps(data))

    def extract(self):
        thread_stop = threading.Event()
        _thread = threading.Thread(
            target=self._publish_stream, args=(thread_stop))
        _thread.start()
        while True:
            try:
                _id = self.image_queue.get()
                result = None
                while result is None:
                    result = self._redis_db.get(_id)

                result = self._decode_redis_data(result)
                result['id'] = _id
                self._redis_db.delete(_id)
                yield result
            except:
                break

        thread_stop.set()
