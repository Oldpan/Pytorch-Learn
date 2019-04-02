import onnx
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image

onnx_model = onnx.load('../../test/new-mobilenetv2-128_S.onnx')

img = Image.open('../../datasets/hand-image/paper.jpg').resize((128, 128))

img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

# target = tvm.target.create('llvm')
#
# input_name = '0'  # change '1' to '0'
# shape_dict = {input_name: x.shape}
# sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#
# with relay.build_config(opt_level=2):
#     graph, lib, params = relay.build_module.build(sym, target, params=params)
#
# dtype = 'float32'

from tvm.contrib import graph_runtime, util


print("Dumping model files...")
libpath = "../tvm_output_lib/mobilenet.so"
# lib.export_library(libpath)

graph_json_path = "../tvm_output_lib/mobilenet.json"
# with open(graph_json_path, 'w') as fo:
#     fo.write(graph)

param_path = "../tvm_output_lib/mobilenet.params"
# with open(param_path, 'wb') as fo:
#     fo.write(relay.save_param_dict(params))


# load the module back.
loaded_json = open(graph_json_path).read()
loaded_lib = tvm.module.load(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

ctx = tvm.cpu()

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input("0", x)
module.run()
out_deploy = module.get_output(0).asnumpy()

print(out_deploy)










# since = time.time()
# for i in range(10000):
#     output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
# time_elapsed = time.time() - since
# print('Time elapsed is {:.0f}m {:.0f}s'.
#       format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间


# from matplotlib import pyplot as plt
# out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode='L')
# out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
# out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
# result = Image.merge('YCbCr', [out_y, out_cb, out_cr]).convert('RGB')
# canvas = np.full((672, 672*2, 3), 255)
# canvas[0:224, 0:224, :] = np.asarray(img)
# canvas[:, 672:, :] = np.asarray(result)
# plt.imshow(canvas.astype(np.uint8))
# plt.show()
