import os
import os.path as osp
import platform
import cv2
import numpy
import matplotlib.pyplot as plt

root_path = osp.dirname(osp.abspath(__file__))
image_path = osp.join(root_path, 'datasets\images')


# return the image path in a folder
def get_image_path(path):
    return [x.path for x in os.scandir(path) if x.name.endswith(".jpg") or x.name.endswith(".png")]


def get_current_version():
    return platform.python_version_tuple()


# load images with arbitrary size
def load_images_r(images_path, transform=None):
    iter_all_image = (cv2.imread(ph) for ph in images_path)
    if transform:
        iter_all_image = (transform(img) for img in iter_all_image)
    all_images = []
    for image in iter_all_image:
        all_images.append(image)
    return all_images


# load images with same size
def load_images(images_path, transform=None):
    iter_all_image = (cv2.imread(ph) for ph in images_path)
    if transform:
        iter_all_image = (transform(img) for img in iter_all_image)
    for i, image in enumerate(iter_all_image):
        if i == 0:
            all_images = numpy.empty((len(images_path),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images


if __name__ == '__main__':
    paths = get_image_path(image_path)
    test = load_images_r(paths)
    test[0] = cv2.cvtColor(test[0], cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(test[0])
    plt.pause(0.001)

    pass
