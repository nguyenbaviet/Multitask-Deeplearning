import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split

class ExamSinglePersion(object):
    """
    Dataloader for pose dataset

    """
    def __init__(self, json_path, image_path, mode, test_size=0.1):
        self.image_path = image_path
        self.database = self.extract_json_file(json_path, mode)
        self.test_size=test_size

    def extract_json_file(self, json_path, mode):
        with open(json_path) as f:
            data = json.load(f)

        database = {}
        img = []
        bbox = []
        pose = []

        for key, d in data.items():
            for id in range(len(d['image'])):
                i_img = d['image'][id]
                i_bbx = np.array(d['bbox'][id], dtype=int)
                i_pose = d['pose'][id]
                i_pose = np.array(i_pose)
                i_pose = np.concatenate((i_pose, np.ones((7, 1))), axis=-1)
                img.append(i_img)
                bbox.append(i_bbx)
                pose.append(i_pose.tolist())

        img_tr, img_te, bbox_tr, bbox_te, pose_tr, pose_te = train_test_split(img, bbox, pose, test_size=self.test_size)

        if mode == 'train':
            img = img_tr
            bbox = bbox_tr
            pose = pose_tr
        else:
            img = img_te
            bbox = bbox_te
            pose = pose_te
        database['image'] = img
        database['bbox'] = bbox
        database['pose'] = pose

        return database

    def get_image(self, img, bbox):
        img = cv2.imread(self.image_path + '/' + img)/255
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = cv2.resize(img, (256, 256))
        return img

    def get_data(self, key, mode=0):
        img = self.database['image'][key]
        bbox = self.database['bbox'][key]
        pose = self.database['pose'][key]
        frame = self.get_image(img, bbox)

        output = {}
        output['frame'] = frame
        output['pose'] = pose
        return output

    def get_length(self, mode=0):
        return len(self.database['image'])
    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return (256, 256, 3)
        if dictkey == 'pose':
            return (7, 3)


class ExamAction(object):
    """
    Dataloader for action dataset
    """""
    def __init__(self, img_path, json_path, clip_size, mode, test_size=0.1, n_class=4):
        self.image_path = img_path
        self.clip_size = clip_size
        self.data = self.extract_json_file(json_path, mode)
        self.test_size=test_size
        self.n_class= n_class

    def extract_json_file(self, json_path, mode):
        with open(json_path) as f:
            data = json.load(f)
        database = {}
        image = []
        label = []
        bbox = []

        for key, d in data.items():
            lb = d['label']
            imgs = d['image']
            bb = np.array(d['bbox'], dtype=int)
            n_frame = len(d['image'])
            id = 0
            for start in range(0, n_frame - self.clip_size, int(self.clip_size / 2)):
                if id == 100:
                    break
                id += 1
                end = start + self.clip_size
                image.append(imgs[start:end])
                bbox.append(bb[start:end])
                label.append(lb)
        img_tr, img_te, bbox_tr, bbox_te, lb_tr, lb_te = train_test_split(image, bbox, label, test_size=self.test_size)

        if mode == 'train':
            bbox = bbox_tr
            image = img_tr
            label = lb_tr
        else:
            bbox = bbox_te
            image = img_te
            label = lb_te

        database['bbox'] = bbox
        database['image'] = image
        database['label'] = label

        return database

    def get_data(self, key, mode=0):
        output = {}
        label = self.data['label'][key]
        lbs = np.zeros(self.n_class)
        lbs[label] = 1
        imgs = self.data['image'][key]
        bboxes = self.data['bbox'][key]
        frame = self.get_video(imgs, bboxes)
        output['frame'] = frame
        output['action'] = lbs

        return output

    def get_video(self, imgs, bboxes):
        video = []
        for id in range(len(imgs)):
            bb = bboxes[id]
            img = cv2.imread(self.image_path + '/' + imgs[id]) / 255
            img = img[bb[1]:bb[3], bb[0]:bb[2]]
            img = cv2.resize(img, (256, 256))
            video.append(img)
        return video

    def get_length(self, mode=0):
        return len(self.data['label'])

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return (self.clip_size, 256, 256, 3)
        if dictkey == 'action':
            return (self.n_class,)