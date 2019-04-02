import imgaug
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('../utils/106553_sat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.title('before augmentation')
plt.imshow(image)
plt.pause(0.001)

transform_sub = imgaug.augmenters.SomeOf((2, None), [
                    imgaug.augmenters.Fliplr(1),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
                    imgaug.augmenters.Flipud(1),
                    imgaug.augmenters.Crop(percent=(0, 0.1)),
                    imgaug.augmenters.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-15, 15),
                        shear=(-4, 4)),
                ], random_order=True)

transform = imgaug.augmenters.Sequential([imgaug.augmenters.Scale(128),
                                          transform_sub])

_, axes = plt.subplots(4, 5, figsize=(30, 20))
for i in range(4):
    # imgaug.seed(i)
    det = transform.to_deterministic()
    imagex = det.augment_image(image)
    axes[i, 0].imshow(imagex)
    # imgaug.seed(i+1)
    det = transform.to_deterministic()
    imagex = det.augment_image(image)
    axes[i, 1].imshow(imagex)

    det = transform.to_deterministic()
    imagex = det.augment_image(image)
    axes[i, 2].imshow(imagex)

    det = transform.to_deterministic()
    imagex = det.augment_image(image)
    axes[i, 3].imshow(imagex)

    det = transform.to_deterministic()
    imagex = det.augment_image(image)
    axes[i, 4].imshow(imagex)

plt.pause(0.01)
