import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

from deephar.config import pennaction_dataconf
from deephar.config import mpii_dataconf
from deephar.config import ModelConfig

from deephar.data import ExamSinglePersion, ExamAction, COCOSinglePerson, MpiiSinglePerson
from deephar.data import BatchLoader

from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))

from exam_tools import ExamActionEvalCallback
from coco_tools import CocoEvalCallback
from mpii_tools import MpiiEvalCallback


num_frames = 14
cfg = ModelConfig((num_frames, ) + pennaction_dataconf.input_shape, pa16j2d,
                  num_actions=[4], num_pyramids=2, action_pyramids=[1, 2],
                  num_levels=4, pose_replica=False,
                  num_pose_features=160, num_visual_features=160)
num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

full_model = spnet.build(cfg)
# full_model.load_weights('/mnt/hdd3tb/Users/hoang/viet/exam/output/nmpii_exam_038.hdf5')

start_lr = 0.001
action_weight = 0.01

batch_clip = 4
# batch_pose = 8

# json_pose_path = '/mnt/hdd3tb/Users/hoang/viet/exam/keypoint_0704.json'
# json_action_path = '/mnt/hdd3tb/Users/hoang/viet/exam/video_2904.json'
# image_path = '/mnt/hdd3tb/Users/hoang/viet/exam/images'
json_pose_path = os.getcwd() + '/datasets/exam/keypoint_0704.json'
json_action_path = os.getcwd() + '/datasets/exam/video_2904.json'
image_path = os.getcwd() + '/datasets/exam/images'


# exam_pose = MpiiSinglePerson('/mnt/hdd3tb/Users/hoang/viet/MPII', dataconf=mpii_dataconf, poselayout=pa16j2d)
exam_pose = MpiiSinglePerson(os.getcwd() + '/datasets/MPII', dataconf=mpii_dataconf, poselayout=pa16j2d)

# exam_pose = COCOSinglePerson("/mnt/hdd3tb/Datasets/COCO2017/images/train2017","/mnt/hdd3tb/Datasets/COCO2017/annotations/person_keypoints_train2017.json")
# exam_pose_te = COCOSinglePerson("/mnt/hdd3tb/Datasets/COCO2017/images/val2017","/mnt/hdd3tb/Datasets/COCO2017/annotations/person_keypoints_val2017.json")

pe_data_tr = BatchLoader([exam_pose], ['frame'], ['pose'], TRAIN_MODE,
                         batch_size= [num_frames], num_predictions=num_predictions, shuffle=True)

#To make the pose data have the same shape with action data
pe_data_tr = BatchLoader(pe_data_tr, ['frame'], ['pose'], TRAIN_MODE,
                         batch_size= batch_clip, num_predictions=num_predictions, shuffle=True)

# pe_data_te = BatchLoader(exam_pose_te, ['frame'], ['pose'], TEST_MODE,
#                          batch_size= exam_pose_te.get_length(), num_predictions=num_predictions, shuffle=True)

exam_action_tr = ExamAction(image_path, json_action_path, num_frames,mode=0)
exam_action_te = ExamAction(image_path, json_action_path, num_frames,mode=1)


ar_data_tr = BatchLoader(exam_action_tr, ['frame'], ['action'], TRAIN_MODE,
                         batch_size=batch_clip, num_predictions=num_action_predictions, shuffle=True)

# save_model = SaveModel('/mnt/hdd3tb/Users/hoang/viet/exam/output/nmpii_exam_{epoch:03d}.hdf5', model_to_save=full_model)
save_model = SaveModel(os.getcwd() + '/output/mpii_exam_{epoch:03d}.hdf5', model_to_save=full_model)

mpii_val = BatchLoader(exam_pose, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=224, shuffle=False)
printnl('Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]
def prepare_training(pose_trainable, lr):
    optimizer = RMSprop(lr=lr)
    models = compile_split_models(full_model, cfg, optimizer,
            pose_trainable=pose_trainable, ar_loss_weights=action_weight,
            copy_replica=cfg.pose_replica)

    """Create validation callbacks."""
    penn_callback = ExamActionEvalCallback(exam_action_te, eval_model=models[1])
    # coco_callback = CocoEvalCallback(pe_data_te, eval_model=models[0])
    mpii_callback = MpiiEvalCallback(x_val, p_val, afmat_val, head_val,
            eval_model=models[0], pred_per_block=1, batch_size=1)
    def end_of_epoch_callback(epoch):

        save_model.on_epoch_end(epoch)
        # coco_callback.on_epoch_end(epoch)
        mpii_callback.on_epoch_end(epoch)
        penn_callback.on_epoch_end(epoch)
        if epoch in [15, 25]:
            lr = float(K.get_value(optimizer.lr))
            newlr = 0.1*lr
            K.set_value(optimizer.lr, newlr)
            printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' \
                    % (lr, newlr, epoch))

    return end_of_epoch_callback, models

# steps_per_epoch = exam_action_tr.get_length(TRAIN_MODE)
steps_per_epoch = exam_pose.get_length(TRAIN_MODE) // num_frames
# steps_per_epoch = 1
fcallback, models = prepare_training(False, start_lr)

# trainer = MultiModelTrainer(models[1:], [ar_data_tr], workers=12,
#         print_full_losses=True)
# trainer.train(2, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        # end_of_epoch_callback=fcallback)

"""Joint learning the full model."""
fcallback, models = prepare_training(True, start_lr)
trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=12,
# trainer = MultiModelTrainer([models[0]], [pe_data_tr], workers=12,
        print_full_losses=True)
trainer.train(30, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        end_of_epoch_callback=fcallback)