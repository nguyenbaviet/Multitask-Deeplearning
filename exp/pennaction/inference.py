import os, sys
sys.path.append(os.getcwd())
from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import numpy as np
import time
plugin_dir = None
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)

net = IENetwork.from_ir(model=os.getcwd() + '/output/penn.xml', weights=os.getcwd()+'/output/penn.bin')
input_blod = next(iter(net.inputs))

exec_net = plugin.load(network=net)
del net

a = cv2.imread(os.getcwd() + '/test_pose.jpg')
a = cv2.resize(a, (256, 256))
i_data = [a, a, a, a, a, a, a, a]
i_data = np.expand_dims(i_data, axis=0)
print(i_data.shape)
i_data = i_data.transpose(0, 4, 1, 2, 3)
start = time.time()
_ = exec_net.infer({input_blod: i_data})
print(time.time() - start)