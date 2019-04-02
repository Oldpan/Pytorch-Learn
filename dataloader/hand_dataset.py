import cv2
import numpy as np

import torch.utils.data as data
from torchvision.transforms import transforms

import os
import glob
from pathlib import Path
import os.path as osp

import warnings

warnings.filterwarnings("ignore")

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def range_of(x):
    """Create a range from 0 to `len(x)`."""
    return list(range(len(x)))


def arange_of(x):
    "Same as `range_of` but returns an array."
    return np.arange(len(x))


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_subsets(dir):
    sets = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return sets


def image_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class HandDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        augment (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
    """

    def __init__(self, root, loader=image_loader, train='train', mode='train', valid_pct=None, augment=None):

        self.root = Path(root)
        self.train = Path(train)
        self.loader = loader
        self.image_files = []
        self.train_files = []
        self.val_files = []
        self.is_valid_pct = 0
        self.mode = mode

        if valid_pct:
            self.is_valid_pct = 1
            self.valid_pct = valid_pct
            self.class_folders = find_subsets(root / train)
        else:
            self.sub_sets = find_subsets(root)

        self.len = 0
        self.aug_count = 5
        self.augment = augment  # intend to use amgaug
        self.to_tensor = transforms.ToTensor()

    def load(self):

        image_class = []

        for i in range(len(self.class_folders)):
            image_paths = glob.glob(osp.join(self.root / self.train / self.class_folders[i], '*.jpg'))
            image_class.append(image_paths)

        for i in range(len(image_class)):
            for j in range(len(image_class[i])):
                image = {
                    'image_id': image_class[i][j],
                    'image_label': self.class_folders[i],
                    'image_label_id': i,
                }
                self.image_files.append(image)

        if self.is_valid_pct:

            rand_idx = np.random.permutation(range_of(self))
            cut = int(self.valid_pct * len(self))

            val_idx = rand_idx[:cut]
            train_idx = np.setdiff1d(arange_of(self.image_files), val_idx)

            self.train_files = [self.image_files[i] for i in train_idx]
            self.val_files = [self.image_files[i] for i in val_idx]

        print('{} images found in {} folder'.format(len(self.image_files), self.train))
        print('{} images in train'.format(len(self.train_files)))
        print('{} images in valid'.format(len(self.val_files)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        if self.mode == 'train':

            image_file = self.train_files[index]
            image = image_loader(image_file['image_id'])
            image_label = image_file['image_label_id']

            if self.augment:
                import imgaug

                MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                   "Fliplr", "Flipud", "CropAndPad",
                                   "Affine", "PiecewiseAffine"]

                def hook(images, augmenter, parents, default):
                    """Determines which augmenters to apply to masks."""
                    return augmenter.__class__.__name__ in MASK_AUGMENTERS

                image_shape = image.shape

                det = self.augment.to_deterministic()
                image = det.augment_image(image)

                assert image.shape == image_shape, "Augmentation shouldn't change image size"

        else:

            image_file = self.val_files[index]
            image = image_loader(image_file['image_id'])
            image_label = image_file['image_label_id']

        return self.to_tensor(image), image_label

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
