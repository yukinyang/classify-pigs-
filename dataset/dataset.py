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



batch_w = 400
batch_h = 300

class TestLoader(Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])
        low.resize([300, 400])

        h = low.shape[0]
        w = low.shape[1]
        print(h, w)
        #
        # h_offset = random.randint(0, max(0, h - batch_h - 1))
        # w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count