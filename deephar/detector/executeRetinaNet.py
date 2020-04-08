import keras
import sys, os
sys.path.append(os.getcwd())


from deephar.detector.keras_retinanet import models
from deephar.detector.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import cv2
import os
import numpy as np
import time
import csv
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('.', 'deephar/detector/inference', 'resnet50_csv_10.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

cap = cv2.VideoCapture(os.getcwd() + '/infer.avi')
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('result_haha.avi', fourcc, 30, (1280, 720))
id = 0
while True:
    _, img = cap.read()
    if img is None:
        break
    if id ==50:
        break
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    new_img = preprocess_image(new_img)
    new_img, scale = resize_image(new_img)
    start = time.time()
    bboxes, scores, labels = model.predict(np.expand_dims(new_img, axis=0))
    print('time: ', time.time() - start)
    for i, bb in enumerate(bboxes[0]):
        score = scores[0][i]
        if score < 0.5:
            continue
        print(score)
        print(bb)
        bb = bb / scale
        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
    # img = cv2.putText(img, str(result), (bb[0], bb[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    # cv2.imshow('hi', img)
    # cv2.waitKey(0)
    id += 1
    out.write(img)
