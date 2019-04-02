import onnx
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image
from tvm import rpc

onnx_model = onnx.load('../../test/new-mobilenetv2-128_S.onnx')

img = Image.open('../../datasets/hand-image/paper.jpg').resize((128, 128))

img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

input_name = '0'  # change '1' to '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

target = tvm.target.arm_cpu('rasp3b')

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(sym, target, params=params)

dtype = 'float32'

from tvm.contrib import graph_runtime, util

temp = util.tempdir()
path = temp.relpath('lib.tar')
lib.export_library(path)

# path_lib = '../tvm_rasp3b/deploy_lib.tar'
# lib.export_library(path_lib)
# with open("../tvm_rasp3b/deploy_graph.json", "w") as fo:
#     fo.write(graph)
# with open("../tvm_rasp3b/deploy_param.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

host = '192.168.1.104'
port = 9000
remote = rpc.connect(host, port)

remote.upload(path)
rlib = remote.load_module('lib.tar')

ctx = remote.cpu()
module = graph_runtime.create(graph, rlib, ctx)
# set parameter (upload params to the remote device. This may take a while)
module.set_input(**params)
# set input data
module.set_input('0', x)
# run
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())

print(top1)

# time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=10)
# cost = time_f(module.run).mean
# print('%g secs/op' % cost)









