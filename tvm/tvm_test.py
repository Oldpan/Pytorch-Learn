import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image

onnx_model = onnx.load('../test/mobilenetv2-64_SS.onnx')

mean = [123., 117., 104.]
std = [58.395, 57.12, 57.375]


def transform_image(image):
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../datasets/hand-image/stone.jpg').resize((64, 64))

# x = transform_image(img)


img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]


# onnx_model = onnx.load('super_resolution_0.2.onnx')
#
# img = Image.open('../datasets/images/plane.jpg').resize((224, 224))
# img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
# img_y, img_cb, img_cr = img_ycbcr.split()
# x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

target = 'llvm'

input_name = '0'  # change '1' to '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# sym = tvm.relay.quantize.quantize(sym, params)

with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

dtype = 'float32'
func = intrp.evaluate(sym)

# for i in range(5):
#
#     output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
#     # tvm_output = intrp.evaluate(sym)(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
#     # tt = tvm_output.argmax()
#     # print(tt)
#     print(output.argmax())


since = time.time()
for i in range(1000):
    output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
time_elapsed = time.time() - since
print('Time elapsed is {:.0f}m {:.0f}s {:.4}ms'.
      format(time_elapsed // 60, time_elapsed % 60, time_elapsed*1000))  # 打印出来时间


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
