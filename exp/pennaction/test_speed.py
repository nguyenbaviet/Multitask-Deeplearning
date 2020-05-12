import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.append(os.getcwd())
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.utils.pose import pa16j2d

from deephar.models import spnet
from deephar.models import split_model

from deephar.data import PennAction
from deephar.data import BatchLoader

from exp.common.penn_tools import eval_singleclip_gt_bbox_generator

cfg = ModelConfig((8,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=2, action_pyramids=[1,2],
        num_levels=4, pose_replica=False,
        num_pose_features=160, num_visual_features=160)

full_model = spnet.build(cfg)
full_model.load_weights(os.getcwd() + '/output/final_inception.hdf5')
model = split_model(full_model, cfg)[1]
print('load weight done')
penn_seq = PennAction('/mnt/hdd3tb/Users/hoang/viet/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=True,
        clip_size=8)

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], 0,
        batch_size=1, shuffle=False)
print('start')
eval_singleclip_gt_bbox_generator(model, penn_te)