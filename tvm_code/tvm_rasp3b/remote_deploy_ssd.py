import onnx
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image
from tvm import rpc

onnx_model = onnx.load('/home/prototype/Desktop/Deep-Learning/Pytorch-Learn/tvm/MobileNetV2-SSDLite.onnx')

img = Image.open('../../datasets/hand-image/paper.jpg').resize((300, 300))

dtype = 'float32'

img = np.array(img).transpose((2, 0, 1)).astype(dtype)
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

input_name = '0'  # change '1' to '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype=dtype)

target = tvm.target.arm_cpu('rasp3b')

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(sym, target, params=params)


path_lib = 'deploy_ssd.so'
lib.export_library(path_lib, cc="/usr/bin/arm-linux-gnueabihf-g++")
with open("deploy_ssd.json", "w") as fo:
    fo.write(graph)
with open("deploy_ssd.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))




