import glob
import os
import torchvision.transforms as transforms
import numpy as np
import torch
import random

from torch.utils.data import Dataset
from PIL import Image
import cv2


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transform_=None):
        self.root = root
        self.dirs = ['/stand/com', '/stand/incom', '/sleep/com', '/sleep/incom', '/strange']
        files1 = sorted(glob.glob(root + self.dirs[0] + "/*.*"))
        files2 = sorted(glob.glob(root + self.dirs[1] + "/*.*"))
        files3 = sorted(glob.glob(root + self.dirs[2] + "/*.*"))
        files4 = sorted(glob.glob(root + self.dirs[3] + "/*.*"))
        files5 = sorted(glob.glob(root + self.dirs[4] + "/*.*"))
        self.files = files1 + files2 + files3 + files4 + files5
        for i in range(10):
            random.shuffle(self.files)
        self.transform_ = transforms.Compose(transform_)
        self.vals = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]

        # for file in self.files:
        #     print(file)

    def __getitem__(self, index):
        # print(index)
        file = self.files[index % len(self.files)]
        image_ = Image.open(file)
        val = [0, 0]

        if image_.mode != "RGB":
            image_ = to_rgb(image_)

        for i in range(5):
            length = len(self.root + self.dirs[i])
            if file[0:length] == self.root + self.dirs[i]:
                val = self.vals[i]
                # break

        item = self.transform_(image_)
        return {'img': item, 'val1': val[0], 'val2': val[1], 'name':file}

    def __len__(self):
        return len(self.files)



class TestDataset(Dataset):
    def __init__(self, root, transform_=None):
        self.root = root
        self.files = sorted(glob.glob(root + "/*.*"))
        self.transform_ = transforms.Compose(transform_)

    def __getitem__(self, index):
        # print(index)
        file = self.files[index % len(self.files)]
        image_ = Image.open(file)
        if image_.mode != "RGB":
            image_ = to_rgb(image_)
        item = self.transform_(image_)
        return {'img': item, 'name':file}

    def __len__(self):
        return len(self.files)