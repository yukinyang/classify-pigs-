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
        ## 32 x 32
        self.conv2 = nn.Conv2d(self.number_f, self.number_f, 3, 2, 1, bias=True)
        ## 16 x 16
        self.conv3 = nn.Conv2d(self.number_f, self.number_f, 3, 2, 1, bias=True)
        ## 8 x 8
        self.lin1 = nn.Linear(8*8*self.number_f, 64)
        # self.lin2 = nn.Linear(64, self.num_classes1)
        # self.lin3 = nn.Linear(64, self.num_classes2)
        self.lin2 = nn.Linear(64, 1)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):
        ## input size = 64 x 64
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x3 = x3.view(x3.size(0), 8*8*self.number_f)
        x4 = self.lin1(x3)
        out1 = self.relu(self.lin2(x4))
        out2 = self.relu(self.lin3(x4))
        out1 = torch.max(out1, dim=1)[1].float().requires_grad_()
        out2 = torch.max(out2, dim=1)[1].float().requires_grad_()
        return out1, out2










