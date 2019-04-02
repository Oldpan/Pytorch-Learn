import imgaug
import tqdm
import matplotlib.pyplot as plt
import cv2
import os

from utils.image_util import get_image_path

transform_sub = imgaug.augmenters.SomeOf((2, None), [
                                imgaug.augmenters.GaussianBlur(sigma=(0.0, 2.0)),
                                imgaug.augmenters.Crop(percent=(0, 0.1)),
                                imgaug.augmenters.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-15, 15),
                                    shear=(-8, 8)),
                            ], random_order=True)

transform = imgaug.augmenters.Sequential([imgaug.augmenters.Scale(48),
                                          transform_sub])
i = 1

root_dir = '/home/prototype/Documents/ttt'
data_dir = os.listdir(root_dir)
for dir in data_dir:
    tmp_path = os.path.join(root_dir, dir)
    if not os.path.isdir(tmp_path):
        pass
    else:
        images_path = get_image_path(tmp_path)
        for image_path in tqdm.tqdm(images_path):
            image = cv2.imread(image_path)
            for j in range(20):
                det = transform.to_deterministic()
                imagex = det.augment_image(image)
                new_image_path = os.path.join(tmp_path, 'new_image_{}.jpg'.format(i))
                cv2.imwrite(new_image_path, imagex)
                i += 1
