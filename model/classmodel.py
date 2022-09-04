import torch
import torch.nn as nn


class Classmodel(nn.Module):
    def __init__(self):
        super(Classmodel, self).__init__()
        self.num_classes1 = 3
        self.num_classes2 = 3
        self.inputsize = 64*64

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.number_f = 32
        ## 64 x 64
        self.conv1 = nn.Conv2d(3, self.number_f, 7, 2, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.number_f)
        ## 32 x 32
        self.conv2 = nn.Conv2d(self.number_f, self.number_f*4, 3, 2, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(self.number_f*4)
        ## 16 x 16
        self.conv3 = nn.Conv2d(self.number_f*4, self.number_f*8, 3, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.number_f*8)
        ## 8 x 8
        self.conv4 = nn.Conv2d(self.number_f*8, self.number_f*8, 3, 1, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(self.number_f*8)
        self.conv5 = nn.Conv2d(self.number_f*8, self.number_f*8, 3, 1, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(self.number_f*8)
        self.conv6 = nn.Conv2d(self.number_f*8, self.number_f, 3, 1, 1, bias=True)
        self.bn6 = nn.BatchNorm2d(self.number_f)
        ## 8 x 8
        self.lin1 = nn.Linear(8*8*self.number_f, 64)
        # self.lin2 = nn.Linear(64, self.num_classes1)
        # self.lin3 = nn.Linear(64, self.num_classes2)
        self.lin2 = nn.Linear(64, 1)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):
        ## input size = 64 x 64
        x1 = self.relu(self.conv1(x))
        x1 = self.bn1(x1)
        x2 = self.relu(self.conv2(x1))
        x2 = self.bn2(x2)
        x3 = self.relu(self.conv3(x2))
        x3 = self.bn3(x3)
        x4 = self.relu(self.conv4(x3))
        x4 = self.bn4(x4)
        x5 = self.relu(self.conv5(x4))
        x5 = self.bn5(x5)
        x6 = self.relu(self.conv6(x3+x5))
        x6 = self.bn6(x6)

        x6 = x6.view(x6.size(0), 8*8*self.number_f)
        x7 = self.relu(self.lin1(x6))
        out1 = self.relu(self.lin2(x7))
        out2 = self.relu(self.lin3(x7))
        out1 = torch.squeeze(out1, dim=1)
        out2 = torch.squeeze(out2, dim=1)
        return out1, out2










