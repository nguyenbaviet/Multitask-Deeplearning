import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import numpy as np
# from deephar.checker.checker import FrameChecker
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.utils.pose import pa16j2d

from deephar.models import split_model
from deephar.models import spnet

from deephar.detector.keras_retinanet import models
from deephar.detector.keras_retinanet.utils.image import preprocess_image, resize_image
import tensorflow as tf
import keras
import time

# LABELS = {'nhin_bai': 0, 'su_dung_tl': 1, 'trao_doi': 2, 'unknown': 3}
LABELS = {0: 'nhin_bai', 1: 'su_dung_tl', 2: 'trao_doi', 3: 'unknown'}

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
class Infer:
    def __init__(self, args):
        self.id = 0
        self.seqs = []
        self.bbox = [0, 0, 1280, 720]
        self.n_frames = args['n_frames']
        keras.backend.tensorflow_backend.set_session(get_session())
        self.detector = models.load_model(os.getcwd() + '/deephar/detector/inference/resnet50_csv_10.h5',
                                          backbone_name='resnet50')
        cfg = ModelConfig((self.n_frames, ) + pennaction_dataconf.input_shape, pa16j2d,
                  num_actions=[4], num_pyramids=2, action_pyramids=[1,2],
                  num_levels=4, pose_replica=False,
                  num_pose_features=160, num_visual_features=160)
        self.action = spnet.build(cfg)
        self.action.load_weights(args['checkpoint'])
        self.action = split_model(self.action, cfg)[1]
        print('load model done')

    def run(self, x):
        d_img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        d_img = preprocess_image(d_img)
        d_img, scale = resize_image(d_img)
        d_bboxes, d_scores, _ = self.detector.predict(np.expand_dims(d_img, axis=0))
        for d_i, bb in enumerate(d_bboxes[0]):
            if d_scores[0][d_i] < 0.5:
                continue
            bb /= scale
            self.bbox = np.array(bb, dtype=int)
        x = x / 255
        x = x[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
        x = cv2.resize(x, (256, 256))

        self.seqs.append(x)

        if len(self.seqs) < self.n_frames:
            return None
        else:
            seq = np.expand_dims(self.seqs, axis=0)
            start = time.time()
            result = self.action.predict(seq)[-1]
            print('time: ', time.time() - start)
            self.seqs.pop(0)
            return result

if __name__ == '__main__':
    args = {
        'n_frames': 14,
        'checkpoint':'/mnt/hdd3tb/Users/hoang/viet/exam/output/0704_exam_006.hdf5'
        # 'checkpoint':os.getcwd() + '/output/f_exam_002.hdf5'
    }
    action = Infer(args)
    cap = cv2.VideoCapture(os.getcwd() + '/infer.avi')
    # cap = cv2.VideoCapture(os.getcwd() + '/vlc-record-2020-03-31-09h50m15s-rtsp___192.168.3.22_live-.avi')
    # cap = cv2.VideoCapture('/home/vietnguyen/new_deephar/datasets/new_v1/trao_doi/vlc-record-2020-03-31-18h28m32s-rtsp___192.168.3.22_live-.avi')
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter('result_14.avi', fourcc, 30, (1280, 720))
    id = 0
    while True:
        _, img = cap.read()
        if img is None:
            break
        result = action.run(img)
        if result is None:
            lb = None
        else:
            lb = LABELS[np.argmax(result)]
        bb = action.bbox
        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        img = cv2.putText(img, str(lb), (bb[0], bb[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        img = cv2.putText(img, str(np.amax(result)), (bb[2], bb[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        # cv2.imshow('hi', img)
        # cv2.waitKey(0)
        id += 1
        out.write(img)

    # cv2.destroyAllWindows()