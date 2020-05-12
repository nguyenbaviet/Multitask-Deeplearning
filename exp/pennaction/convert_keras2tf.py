import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import keras.backend as K
import sys, os
sys.path.append(os.getcwd())
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.utils.pose import pa16j2d

from deephar.models import spnet
from deephar.models import split_model

tf.keras.backend.clear_session()

save_pb_dir = os.getcwd() + '/output/'
model_fname = os.getcwd() + '/output/final_inception.hdf5'

cfg = ModelConfig((8,) + pennaction_dataconf.input_shape, pa16j2d,
                  num_actions=[15], num_pyramids=2, action_pyramids=[1, 2],
                  num_levels=4, pose_replica=False,
                  num_pose_features=160, num_visual_features=160)
tf.keras.backend.set_learning_phase(0)
model = spnet.build(cfg)
model.load_weights(model_fname)
model = split_model(model, cfg)[1]
def freeze_graph(graph, session, output, save_pb_dir, save_pb_name, save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

init_op = tf.global_variables_initializer()
session = tf.Session()
session.run(init_op)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs][0:1],
                            save_pb_name='penn.pb', save_pb_dir=save_pb_dir)
