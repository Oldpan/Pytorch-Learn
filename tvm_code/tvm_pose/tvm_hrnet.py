import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image

onnx_model = onnx.load('/Users/oldpan/Documents/Kaggle/simple-HRNet/HRNetw32_256x192.onnx')

mean = [123., 117., 104.]
std = [58.395, 57.12, 57.375]


def transform_image(image):
    # image = image - np.array(mean)
    # image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../../datasets/images/body1.jpeg').resize((192, 256))


img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]


target = 'llvm'

input_name = 'input.1'  # change '1' to '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# sym = tvm.relay.quantize.quantize(sym, params)

with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

dtype = 'float32'
func = intrp.evaluate(sym)

since = time.time()
for i in range(1000):
    output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
time_elapsed = time.time() - since
print('Time elapsed is {:.0f}m {:.0f}s {:.4}ms'.
      format(time_elapsed // 60, time_elapsed % 60, time_elapsed*1000))  # 打印出来时间



