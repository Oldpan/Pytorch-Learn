import time

import torch
import cv2
from tqdm import tqdm
from fastai import *
from fastai.vision import *
from models.MobileNetv2 import mobilenetv2
from dataloader.dataloader_webcam import WebcamLoader, GestureLoader, DataWriter
from utils.fn import getTime

hand_path = Path('/home/prototype/Documents/gesture/hand-images')
save_path = Path('./examples')

tfms = get_transforms(flip_vert=True, max_lighting=0.2, max_zoom=1.1, max_warp=0.05, max_rotate=30)
# 480 270
data = ImageDataBunch.from_folder(hand_path, 'train', valid_pct=0.2, ds_tfms=tfms, size=224, bs=64)

learn = create_cnn(data, mobilenetv2, pretrained=False, metrics=accuracy, callback_fns=ShowGraph)


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == '__main__':

    webcam = 0

    # Load input video
    data_loader = WebcamLoader(webcam, batchSize=1).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading Pretrained model..')
    sys.stdout.flush()
    learn.load('new-mobilenetv2-128_S')
    Gesture_loader = GestureLoader(data_loader, learn.model, batchSize=1).start()

    writer = DataWriter(False, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
    batchSize = 25
    for i in im_names_desc:
        try:
            with torch.no_grad():
                predict, orig_img, im_name = Gesture_loader.read()
                writer.save(predict, orig_img, im_name.split('/')[-1])

        except KeyboardInterrupt:
            break

    print(' ')
    print('===========================> Finish Model Running.')
