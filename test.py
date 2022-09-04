from model.classmodel import *
from dataset.dataset import *

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.autograd import Function


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=12)
    parser.add_argument("--data_path", type=str, default='./test imgs/')
    # parser.add_argument("--data_path", type=str, default='./imgs/')
    parser.add_argument("--img_size", type=int, default=[64, 64])

    print(parser.parse_args())
    return parser.parse_args()


def Test_train():
    opt = getparser()

    model = Classmodel()
    checkpoint = torch.load('./save_models/200_classify.pth')
    model.load_state_dict(checkpoint['classify'])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()

    transforms_ = [
        transforms.Resize((opt.img_size[0], opt.img_size[1]), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    dataloader = DataLoader(
        TestDataset(opt.data_path, transform_=transforms_),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpus,
    )
    now = 0
    numbatches = len(dataloader)
    model.eval()
    classes = [
        ['stand', 'sleep', 'strange'],
        ['com', 'incom']
    ]
    for epoch in range(0, 1):
        pbar = enumerate(dataloader)
        nowloss = 0
        for i, batch in pbar:
            # set model input
            input = Variable(batch['img'].type(Tensor))
            input_name = batch['name']
            out1, out2 = model(input)
            [c1], [c2] = torch.round(out1).cpu().detach().numpy(), torch.round(out2).cpu().detach().numpy()
            result = classes[0][int(c1)]
            if c2 == 0 or c2 == 1:
                result = result + ' & ' + classes[1][int(c2)]
            now += 1
            print("Test sample:", str(now))
            print("File name:", input_name)
            print("Result:", result)
        print("======== test has been finished ========")


if __name__ == '__main__':
    Test_train()








