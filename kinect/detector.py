import os
import sys
import time
import json
import base64
import numpy as np
import redis
import tensorflow as tf
import cv2

import keras
from retinanet import get_model, preprocess_image, resize_image, labels_to_names
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())

REDIS_DB = redis.StrictRedis(host='localhost', port='6379', db='0')
REDIS_QUEUE = 'image_queue'
BATCH_SIZE = 32


def _base64_encode(a):
    a = a.copy(order='C')
    return base64.b64encode(a).decode("utf-8")


def _base64_decode(a, shape=None):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    a = np.frombuffer(base64.decodestring(a), dtype=np.uint8)
    if shape is not None:
        a = a.reshape(shape)
    return a


def main_batch():
    primary_model = get_model()
    while True:
        time.sleep(0.01)
        _queue = REDIS_DB.lrange(REDIS_QUEUE, 0, BATCH_SIZE - 1)
        image_queue = []
        batch = None
        for _q in _queue:
            _q = json.loads(_q.decode("utf-8"))
            img = _base64_decode(_q['image'], _q['shape'])

            draw = img.copy()
            # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            img = preprocess_image(img)
            img, scale = resize_image(img)
            img = np.expand_dims(img, axis=0)
            image_queue.append((scale, draw, _q['id']))
            if batch is None:
                batch = img
            else:
                batch = np.vstack([batch, img])
        if image_queue:
            print(batch.shape)
            boxes, scores, labels = primary_model.predict_on_batch(batch)
            for i in range(boxes.shape[0]):
                scale, draw, _id = image_queue[i]
                all_label = []
                for box, score, label in zip(boxes[i], scores[i], labels[i]):
                    if score < 0.5:
                        break
                    box /= scale
                    b = box.astype(int)
                    color = label_color(label)
                    draw_box(draw, b, color=color)
                    caption = "{} {:.3f}".format(labels_to_names(label), score)
                    all_label.append(labels_to_names(label))
                    draw_caption(draw, b, caption)
                data = {
                    'image': _base64_encode(draw),
                    'labels': all_label
                }
                REDIS_DB.set(_id, json.dumps(data))
            REDIS_DB.ltrim(REDIS_QUEUE, len(_queue), -1)

def main():
    primary_model = get_model()
    while True:
        time.sleep(0.01)
        _queue = REDIS_DB.lrange(REDIS_QUEUE, 0, BATCH_SIZE - 1)
        for _q in _queue:
            _q = json.loads(_q.decode("utf-8"))
            img = _base64_decode(_q['image'], _q['shape'])

            draw = img.copy()
            # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            img = preprocess_image(img)
            img, scale = resize_image(img)
            img = np.expand_dims(img, axis=0)
            boxes, scores, labels = primary_model.predict_on_batch(img)
            boxes /= scale
            all_label = []
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < 0.5:
                    break    
                b = box.astype(int)
                color = label_color(label)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names(label), score)
                all_label.append(labels_to_names(label))
                draw_caption(draw, b, caption)
            data = {
                'image': _base64_encode(draw),
                'labels': all_label
            }
            REDIS_DB.set(_q['id'], json.dumps(data))
        REDIS_DB.ltrim(REDIS_QUEUE, len(_queue), -1)


if __name__ == "__main__":
    print("Running detector")
    main()
