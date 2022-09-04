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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_cpus", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, default='./img/')
    # parser.add_argument("--data_path", type=str, default='./imgs/')
    parser.add_argument("--img_size", type=int, default=[64, 64])
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


def Test_train():
    opt = getparser()

    model = Classmodel()
    # checkpoint = torch.load('./save/200Guidemap_small.pth')
    # guidemap_model.load_state_dict(checkpoint['Guide'])

    LOSS = nn.L1Loss()

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
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    # run_dir = get_dir_name('./run', 'DecomRLTrain')
    # os.makedirs(run_dir)
    # os.makedirs(run_dir + '/save_files')
    # shutil.copyfile('./util/GuidemapLoss.py', run_dir + '/save_files/' + 'GuidemapLoss.py')
    # shutil.copyfile('./model/guidemodel.py', run_dir + '/save_files/' + 'guidemodel.py')
    # for epoch in range(0, opt.epochs):
    for epoch in range(0, 1):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=numbatches)
        nowloss = 0
        for i, batch in pbar:
            # set model input
            input = Variable(batch['img'].type(Tensor))
            Trueval1 = Variable(batch['val1'].type(Tensor)).cuda()
            Trueval2 = Variable(batch['val2'].type(Tensor)).cuda()
            print(Trueval1.shape)
            print(Trueval2.shape)

            # Train
            model.train()
            optimizer.zero_grad()

            # DecomRL map
            out1, out2 = model(input)
            # out1 = torch.max(out1, dim=1)[1].float().requires_grad_()
            # out2 = torch.max(out2, dim=1)[1].float().requires_grad_()
            print(out1.shape)
            print(out2.shape)
            break
            Loss = LOSS(out1, Trueval1) + LOSS(out2, Trueval2)
            # Loss.float().requires_grad_()
            # print(Loss)

            nowloss = nowloss + Loss
            Loss.backward()
            optimizer.step()
            now += 1

        lr_scheduler.step()
    #
    #     # save model
    #     if (epoch >= 399 and (epoch + 1) % 100 == 0) or epoch == 1:
    #         model_guidemap_path = run_dir + '/save_files/' + str(epoch + 1) + 'DecomRL.pth'
    #         torch.save({'Decom':model.state_dict()}, model_guidemap_path)
    #
        nowloss = nowloss / numbatches
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")


if __name__ == '__main__':
    Test_train()








