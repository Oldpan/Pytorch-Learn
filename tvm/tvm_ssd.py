import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_runtime
from PIL import Image

onnx_model = onnx.load('../tvm/MobileNetV2-SSDLite.onnx')

mean = [123., 117., 104.]
std = [58.395, 57.12, 57.375]


def transform_image(image):
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../datasets/images/boy.jpg').resize((300, 300))

# x = transform_image(img)


img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0                       # remember pytorch tensor is 0-1
x = img[np.newaxis, :]


# target = 'llvm'
#
# input_name = '0'
# shape_dict = {input_name: x.shape}
# sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#
# with relay.build_config(opt_level=3):
#     graph, lib, params = relay.build_module.build(sym, target, params=params)

# with relay.build_config(opt_level=3):
#     intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

print("Reading model files...")
libpath = "/home/prototype/Desktop/Deep-Learning/Pytorch-Learn/tvm/tvm_quantize/MobileNetV2-SSDLite.so"
# lib.export_library(libpath)

graph_json_path = "/home/prototype/Desktop/Deep-Learning/Pytorch-Learn/tvm/tvm_quantize/MobileNetV2-SSDLite.json"
# with open(graph_json_path, 'w') as fo:
#     fo.write(graph)

param_path = "/home/prototype/Desktop/Deep-Learning/Pytorch-Learn/tvm/tvm_quantize/MobileNetV2-SSDLite.params"
# with open(param_path, 'wb') as fo:
#     fo.write(relay.save_param_dict(params))

dtype = 'float32'

loaded_json = open(graph_json_path).read()
loaded_lib = tvm.module.load(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

ctx = tvm.cpu()

m = graph_runtime.create(loaded_json, loaded_lib, ctx)
m.set_input('0', tvm.nd.array(x.astype(dtype)))
m.load_params(loaded_params)

m.run()
tvm_output = m.get_output(0).asnumpy()

print(tvm_output.shape)

since = time.time()
for i in range(1000):
    m.run()
time_elapsed = time.time() - since
print('Time elapsed is {:.0f}m {:.0f}s {:.4}ms'.
      format(time_elapsed // 60, time_elapsed % 60, time_elapsed*1000))  # 打印出来时间





