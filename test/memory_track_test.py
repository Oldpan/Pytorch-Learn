import torch
import torch.nn as nn
import inspect

from torchvision import models
from utils.gpu_memory_util.gpu_mem_track import MemTracker

device = torch.device('cuda:0')

frame = inspect.currentframe()
gpu_tracker = MemTracker(frame)

# gpu_tracker.track()
# cnn = models.vgg19(pretrained=True).to(device)
gpu_tracker.track()


dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1000/1000 = 94.37M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1000/1000 = 125.82M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1000/1000 = 188.74M

gpu_tracker.track()
#
# dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1000/1000 = 377.48M
# dummy_tensor_5 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1000/1000 = 251.64M
#
# gpu_tracker.track()
#
# dummy_tensor_4 = dummy_tensor_4.cpu()
# dummy_tensor_2 = dummy_tensor_2.cpu()
# torch.cuda.empty_cache()

gpu_tracker.track()

layers = ['relu_1', 'relu_3', 'relu_5', 'relu_9']
layerIdx = 0

content_image = torch.randn(1, 3, 500, 500).float().to(device)
style_image = torch.randn(1, 3, 500, 500).float().to(device)
feature_extractor = nn.Sequential().to(device)
cnn = models.vgg19(pretrained=True).features.to(device)
gpu_tracker.track()

input_features = []
target_features = []
i = 0
with torch.no_grad():
    for layer in cnn.children():

        if layerIdx < len(layers):
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_" + str(i)
                feature_extractor.add_module(name, layer)

            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                feature_extractor.add_module(name, layer)

            elif isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                feature_extractor.add_module(name, nn.ReLU(inplace=True))

            if name == layers[layerIdx]:
                input = feature_extractor(content_image)

                target = feature_extractor(style_image)

                input_features.append(input)
                target_features.append(target)

                layerIdx += 1

#
# for buf in feature_extractor.buffers():
#     print(type(buf.data), buf.size())

# torch.cuda.empty_cache()

gpu_tracker.track()

# del cnn
del feature_extractor

torch.cuda.empty_cache()
gpu_tracker.track()

del cnn
torch.cuda.empty_cache()
gpu_tracker.track()