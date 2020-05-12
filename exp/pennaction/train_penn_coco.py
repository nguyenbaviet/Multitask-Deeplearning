import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

from deephar.config import mpii_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.data import COCOSinglePerson
from deephar.data import PennAction
from deephar.data import BatchLoader

from keras.optimizers import RMSprop
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
# from datasetpath import datasetpath

from exp.common.coco_tools import CocoEvalCallback
from penn_tools import PennActionEvalCallback

logdir = '/home/vietnguyen/new_deephar/output/'
# if len(sys.argv) > 1:
#     logdir = sys.argv[1]
#     mkdir(logdir)
#     sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.001
action_weight = 0.01
batch_size_mpii = int(0.8 * num_frames)
# batch_size_penn = num_frames - batch_size_mpii
batch_size_penn = num_frames
batch_clips = 4# 8/4

"""Load datasets"""
coco = COCOSinglePerson("/mnt/hdd3tb/Datasets/COCO2017/images/train2017/",
                        "/mnt/hdd3tb/Datasets/COCO2017/annotations/person_keypoints_train2017.json",
                        dataconf=mpii_dataconf)
coco_val = COCOSinglePerson("/mnt/hdd3tb/Datasets/COCO2017/images/val2017/",
                        "/mnt/hdd3tb/Datasets/COCO2017/annotations/person_keypoints_val2017.json",
                        dataconf=mpii_dataconf)
# penn_sf = PennAction('/mnt/hdd3tb/Users/hoang/viet/PennAction', pennaction_pe_dataconf,
#         poselayout=pa16j2d, topology='frames', use_gt_bbox=True)
penn_seq = PennAction('/mnt/hdd3tb/Users/hoang/viet/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=True,
        clip_size=num_frames)

# pe_data_tr = BatchLoader([mpii, penn_sf], ['frame'], ['pose'], TRAIN_MODE,
pe_data_tr = BatchLoader([coco], ['image'], ['pose'], TRAIN_MODE,
        # batch_size=[batch_size_mpii, batch_size_penn], shuffle=True)
        batch_size=[batch_size_penn], shuffle=True)
pe_data_tr = BatchLoader(pe_data_tr, ['image'], ['pose'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_predictions, shuffle=False)

ar_data_tr = BatchLoader(penn_seq, ['frame'], ['pennaction'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_action_predictions,
        shuffle=True)

"""Build the full model"""
full_model = spnet.build(cfg)
# full_model = multi_gpu_model(full_model0, 2)

"""Load pre-trained weights from pose estimation and copy replica layers."""
# full_model.load_weights(
#         # 'output/mpii_spnet_51_f47147e/weights_mpii_spnet_8b4l_039.hdf5',
#         '/home/vietnguyen/new_deephar/output/inception_003.hdf5',
#         by_name=True)


"""Origin comment"""
# from keras.models import Model
# full_model = Model(full_model.input,
        # [full_model.outputs[5], full_model.outputs[11]], name=full_model.name)
# cfg.num_pyramids = 1
# cfg.num_levels = 2
# cfg.action_pyramids = [2]
# mpii.get_length(VALID_MODE)
"""Trick to pre-load validation samples and generate the eval. callback."""

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Save model callback."""
save_model = SaveModel('/home/vietnguyen/new_deephar/output/pennCoco_{epoch:03d}.hdf5', model_to_save=full_model)


def prepare_training(pose_trainable, lr):
    optimizer = RMSprop(lr=lr)
    models = compile_split_models(full_model, cfg, optimizer,
            pose_trainable=pose_trainable, ar_loss_weights=action_weight,
            copy_replica=cfg.pose_replica)
    full_model.summary()

    """Create validation callbacks."""
    mpii_callback = CocoEvalCallback(eval_model=models[0], datagen=coco_val)
    penn_callback = PennActionEvalCallback(penn_te, eval_model=models[1],
            logdir=logdir)

    def end_of_epoch_callback(epoch):

        save_model.on_epoch_end(epoch)
        mpii_callback.on_epoch_end(epoch)
        penn_callback.on_epoch_end(epoch)

        if epoch in [15, 25]:
            lr = float(K.get_value(optimizer.lr))
            newlr = 0.1*lr
            K.set_value(optimizer.lr, newlr)
            printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' \
                    % (lr, newlr, epoch))

    return end_of_epoch_callback, models

steps_per_epoch = coco.get_length(TRAIN_MODE) // num_frames

fcallback, models = prepare_training(False, start_lr)

"""Viet test"""
# models[0].load_weights('/home/vietnguyen/deephar/weight_coco/final_coco_020.h5')
# print('load weight done')

"""end"""
# trainer = MultiModelTrainer(models[1:], [ar_data_tr], workers=12,
#         print_full_losses=True)
# trainer.train(2, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        # end_of_epoch_callback=fcallback)

"""Joint learning the full model."""
fcallback, models = prepare_training(True, start_lr)
trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=12,
        print_full_losses=True)
trainer.train(20, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        end_of_epoch_callback=fcallback)


