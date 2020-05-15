import onnx
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image
from tvm import rpc

onnx_model = onnx.load('../../test/mobilenetv2-64_SS.onnx')

img = Image.open('../../datasets/hand-image/paper.jpg').resize((64, 64))

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


path_lib = 'deploy_lib_S.so'
lib.export_library(path_lib, cc="/usr/bin/arm-linux-gnueabihf-g++")
with open("deploy_graph_S.json", "w") as fo:
    fo.write(graph)
with open("deploy_param_S.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))











