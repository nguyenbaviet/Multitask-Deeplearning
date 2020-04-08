import sys, os
sys.path.append(os.getcwd())

import cv2
import json
import numpy as np

from deephar.models.reception import build
from deephar.config import mpii_dataconf
import tensorflow as tf
import keras

from deephar.detector.keras_retinanet import models
from deephar.detector.keras_retinanet.utils.image import preprocess_image, resize_image

BBOXES = [[[660, 210, 1030, 490], [500, 200, 960, 510]],
          [[620, 185, 940, 490], [605, 165, 950, 510]],
          [[585, 206, 1130, 620], [380, 200, 1000, 620]],
          [[600, 185, 1000, 530], [600, 185, 1000, 530]]]
D_BBOX = [410, 160, 1150, 650]
LABELS = {'nhin_bai': 0, 'su_dung_tl': 1, 'trao_doi': 2, 'unknown': 3}
# for detector model
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
class ExtractVideo(object):
    """
    Extract video to get pose data
    Use keras-retinanet to predict bbox
    Model to predict pose was train with coco dataset with 7 joints: noise, shoulder left-right, elbow left-right, wrist left-right
    The folder which contain video has named with corresponding label
    input: base_link: path to the folder which contains video
           save_path: path to folder which save both json file and images folder
    """
    def __init__(self, base_link, save_path, bbox=[0, 0, 1280, 720]):
        keras.backend.tensorflow_backend.set_session(get_session())
        self.bbox = bbox
        self.save_path = save_path
        self.base_link = base_link
        self.pose = build(mpii_dataconf.input_shape, num_blocks=7, num_joints=7,
                          dim=2, num_context_per_joint=2, ksize=(5, 5))
        self.pose.load_weights(os.getcwd() + '/output/weight_coco/final_coco_020.h5')
        self.detector = models.load_model(os.getcwd() + '/deephar/detector/inference/resnet50_csv_10.h5',
                                          backbone_name='resnet50')
        print('load weight done')
    def extract(self):
        folders = os.listdir(self.base_link)
        folders.sort()

        database = {}
        for id_f in range(len(folders)):
            label = LABELS[folders[id_f]]
            folder = base_link + '/' + folders[id_f]
            vid = [folder + '/' + v for v in os.listdir(folder)]
            vid.sort()
            for i, v in enumerate(vid):
                print('id: %d  i: %d' %(id_f, i))
                or_bbox = D_BBOX
                name = v.split('/')[-1].split('.')[0]
                data = {}
                bboxes = []
                image = []
                pose = []
                cap = cv2.VideoCapture(v)
                id = 0
                while True:
                    bol, img = cap.read()
                    if img is None:
                        break
                    d_img = img[or_bbox[1]:or_bbox[3], or_bbox[0]:or_bbox[2]]
                    d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                    d_img = preprocess_image(d_img)
                    d_img, scale = resize_image(d_img)
                    d_bboxes, d_scores, d_labels = self.detector.predict(np.expand_dims(d_img, axis=0))
                    self.bbox = None
                    for d_i, bb in enumerate(d_bboxes[0]):
                        score = d_scores[0][d_i]
                        if score < 0.5:
                            continue
                        bb /= scale
                        bb[0::2] += D_BBOX[0]
                        bb[1::2] += D_BBOX[1]
                        bb = np.array(bb, dtype=int) + 10
                        bb = bb.tolist()
                        self.bbox = bb
                    if self.bbox is None:
                        continue
                    i_name = name + '_%04d' % id + '.jpg'
                    id += 1
                    # cv2.imwrite(self.save_path + '/images/' + i_name, img)
                    image.append(i_name)
                    bboxes.append(self.bbox)
                    i_img = img[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]] / 255
                    i_img = cv2.resize(i_img, (256, 256))
                    i_img = np.expand_dims(i_img, axis=0)
                    kp = self.pose.predict(i_img)[-1][0]
                    kp = kp[:, 0:2]
                    pose.append(kp.tolist())
                data['image'] = image
                data['bbox'] = bboxes
                data['pose'] = pose
                data['label'] = label
                database[name] = data
        print('prepare for dump data')
        with open(save_path + '/keypoint_0704.json', 'w') as f:
            json.dump(database, f)
        return database

class ExtractVideo1(object):
    """
    Extract video to get action dataset
    Use keras-restinanet model to predict bbox
    The folder which contain video has named with corresponding label
    Input: base_link: path to the folder which contains video
           save_path: path to folder which save both json file and images folder
    """
    def __init__(self, base_link, save_path, bbox=[0, 0, 1280, 720]):
        self.bbox = bbox
        self.save_path = save_path
        self.base_link = base_link
        keras.backend.tensorflow_backend.set_session(get_session())
        self.detector = models.load_model(os.getcwd() + '/deephar/detector/inference/resnet50_csv_10.h5', backbone_name='resnet50')

    def extract(self):
        folders = os.listdir(self.base_link)
        folders.sort()

        database = {}
        for id_f in range(len(folders)):
            label = LABELS[folders[id_f]]
            folder = base_link + '/' + folders[id_f]
            vid = [folder + '/' + v for v in os.listdir(folder)]
            vid.sort()
            for i, v in enumerate(vid):
                print('id: %d  i: %d' %(id_f, i))
                name = v.split('/')[-1].split('.')[0]
                data = {}
                bboxes = []
                image = []
                cap = cv2.VideoCapture(v)
                id = 0
                while True:
                    bol, img = cap.read()
                    if img is None:
                        break
                    d_img = img[D_BBOX[1]:D_BBOX[3], D_BBOX[0]:D_BBOX[2]]
                    d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                    d_img = preprocess_image(d_img)
                    d_img, scale = resize_image(d_img)
                    d_bboxes, d_scores, d_labels = self.detector.predict(np.expand_dims(d_img, axis=0))
                    self.bbox = D_BBOX
                    for d_i, bb in enumerate(d_bboxes[0]):
                        score = d_scores[0][d_i]
                        if score < 0.5:
                            continue
                        bb /= scale
                        bb[0::2] += D_BBOX[0]
                        bb[1::2] += D_BBOX[1]
                        bb = bb.tolist()
                        self.bbox = bb
                    i_name = name + '_%04d' % id + '.jpg'
                    id += 1
                    cv2.imwrite(self.save_path + '/images/' + i_name, img)
                    image.append(i_name)
                    bboxes.append(self.bbox)
                data['image'] = image
                data['bbox'] = bboxes
                data['label'] = label
                database[name] = data
        print('prepare for dump data')
        with open(save_path + '/extract_0704.json', 'w') as f:
            json.dump(database, f)
        return database

if __name__ == '__main__':
    save_path = '/mnt/hdd3tb/Users/hoang/viet/exam'
    base_link = '/home/vietnguyen/new_deephar/datasets/06_04'
    # extractor = ExtractVideo(base_link, save_path)
    # extractor.extract()

    extractor = ExtractVideo1(base_link, save_path)
    extractor.extract()
    # vid = '/home/nbviet/Videos/infer.avi'
    # cap = cv2.VideoCapture(vid)
    # _, img = cap.read()
    # pose = build(mpii_dataconf.input_shape, num_blocks=7, num_joints=7,
    #                       dim=2, num_context_per_joint=2, ksize=(5, 5))
    # pose.load_weights(os.getcwd() + '/output/weight_coco/final_coco_020.h5')
    #
    # # bbox = [330, 190, 1120, 530]
    # bbox = [600, 190, 910, 530]
    # i_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] / 255
    # i_img = cv2.resize(i_img, (256, 256))
    # i_img = np.expand_dims(i_img, axis=0)
    # kps = pose.predict(i_img)[-1][0]
    # for kp in kps:
    #     x = int(kp[0] * (bbox[2] - bbox[0])) + bbox[0]
    #     y = int(kp[1] * (bbox[3] - bbox[1])) + bbox[1]
    #     img = cv2.circle(img, center=(x, y), radius=3, thickness=3, color=(0, 0, 255))
    # cv2.imshow('hi', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # id = 0
    # while True:
    #     _, img = cap.read()
    #     id += 1
    #     cv2.imshow('hi', img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()