from fastai.vision import *
from fastai import *
from models.MobileNetv2 import mobilenetv2

hand_path = Path('/home/prototype/Documents/gesture/hand-images')
tfms = get_transforms(flip_vert=True, max_lighting=0.6, max_zoom=0.5, max_warp=0.1, max_rotate=30)
data = ImageDataBunch.from_folder(hand_path, 'train', valid_pct=0.2, ds_tfms=tfms, size=128, bs=64)

learn = create_cnn(data, mobilenetv2, pretrained=False, metrics=accuracy)
learn.load('new-mobilenetv2-128_S')
print(learn.model)

example = torch.rand(1, 3, 128, 128)

torch_out = torch.onnx.export(learn.model.cpu(),
                              example,
                              "new-mobilenetv2-128_S.onnx",
                              verbose=True,
                              export_params=True
                              )
