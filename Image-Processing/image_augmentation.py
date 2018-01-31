# Image augmentation,with it we can improve the quality of image processing
# 更多的图像变换应该参考keras->preprocessing->image.py
# 一般的图像数据集增强有：缩放、旋转、翻转、去中心化、取PCA、左右偏置

import numpy
import cv2
from utils import get_image_path, load_images_r, image_path
import matplotlib.pyplot as plt


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
	h, w = image.shape[0:2]
	rotation = numpy.random.uniform(-rotation_range, rotation_range)
	scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
	tx = numpy.random.uniform(-shift_range, shift_range) * w
	ty = numpy.random.uniform(-shift_range, shift_range) * h
	mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
	mat[:, 2] += (tx, ty)
	result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
	if numpy.random.random() < random_flip:
		result = result[:, ::-1]
	return result


if __name__ == '__main__':

	paths = get_image_path(image_path)
	test = load_images_r(paths)
	test[0] = cv2.cvtColor(test[0], cv2.COLOR_BGR2RGB)
	plt.figure()
	plt.imshow(test[0])
	plt.pause(0.001)
	new = random_transform(test[0], 100, 0.5, 0, 0.5)
	plt.figure()
	plt.imshow(new)
	plt.pause(0.001)

	pass



