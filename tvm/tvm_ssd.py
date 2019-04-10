import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
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


img = Image.open('../datasets/hand-image/scissor.jpg').resize((300, 300))

# x = transform_image(img)


img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]


target = 'llvm'

input_name = '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# sym = tvm.relay.quantize.quantize(sym, params)

with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

dtype = 'float32'
func = intrp.evaluate(sym)

output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()

print(output)



