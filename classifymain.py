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


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_cpus", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='./img/')
    # parser.add_argument("--data_path", type=str, default='./imgs/')
    parser.add_argument("--img_size", type=int, default=[64, 64])

    print(parser.parse_args())
    return parser.parse_args()


def Test_train():
    opt = getparser()

    model = Classmodel()

    LOSS = nn.MSELoss()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()
        LOSS.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 80, 2000], gamma=0.1, last_epoch=-1
    )

    transforms_ = [
        transforms.Resize((opt.img_size[0], opt.img_size[1]), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    dataloader = DataLoader(
        ImageDataset(opt.data_path, transform_=transforms_),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpus,
    )
    # print(len(dataloader))

    now = 0
    numbatches = len(dataloader)

    for epoch in range(0, opt.epochs):
    # for epoch in range(0, 1):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=numbatches)
        nowloss = 0
        for i, batch in pbar:
            # set model input
            input = Variable(batch['img'].type(Tensor))
            Trueval1 = Variable(batch['val1'].type(Tensor)).cuda()
            Trueval2 = Variable(batch['val2'].type(Tensor)).cuda()
            # print(Trueval1.shape)
            # print(Trueval2.shape)

            # Train
            model.train()
            optimizer.zero_grad()

            out1, out2 = model(input)
            # print(out1.shape)
            # print(out2.shape)
            Loss = LOSS(out1, Trueval1) + LOSS(out2, Trueval2)

            nowloss = nowloss + Loss
            Loss.backward()
            optimizer.step()
            now += 1

        lr_scheduler.step()
    
        # save model
        if (epoch >= 99 and (epoch + 1) % 100 == 0) or epoch == 1:
            model_guidemap_path = './save_models/' + str(epoch + 1) + '_classify.pth'
            torch.save({'classify':model.state_dict()}, model_guidemap_path)
    
        nowloss = nowloss / numbatches
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")


if __name__ == '__main__':
    Test_train()








