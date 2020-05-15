import os
import time
import tvm
import numpy as np
from PIL import Image
from tvm import relay
import tflite.Model as tflite

tflite_model_file = "/Users/oldpan/Downloads/model.tflite"
# tflite_model_file = "/Users/oldpan/Downloads/multi_person_mobilenet_v1_075_float.tflite"
tflite_model_buf = open(tflite_model_file, "rb").read()

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

H = 192
W = 192

# mean = [12., 117., 104.]
# std = [58.395, 57.12, 57.375]

def transform_image(image):
    # image = image - np.array(mean)
    # image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../datasets/images/body1.jpeg').resize((W,H))

img = np.array(img).astype('float32')

# img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img / 255.0  # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

input_tensor = "image"
input_shape = (1, H, W, 3)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                  relay.transform.ConvertLayout('NCHW')])

with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

target = "llvm"
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

from tvm.contrib import graph_runtime as runtime

# Create a runtime executor module
module = runtime.create(graph, lib, tvm.cpu())

# Feed input data
module.set_input(input_tensor, x)

# Feed related params
module.set_input(**params)


print('Start benchmark...')
since = time.time()
for i in range(1000):
    module.run()
time_elapsed = time.time() - since
print('Total time elapsed is {:.0f}m {:.0f}s ,each {:.4}ms'.
      format(time_elapsed // 60, time_elapsed % 60, time_elapsed))  # 打印出来时间
# Run
# module.run()

# Get output
tvm_output = module.get_output(0).asnumpy()

pass