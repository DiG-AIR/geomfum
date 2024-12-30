
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):

    def __init__(self, global_feat = True, feature_transform = False):

        super(PointNetfeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.dense1 = torch.nn.Linear(1024,256)
        self.dense2 = torch.nn.Linear(256,256)

        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):

        n_pts = x.size()[2]
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv41(x)))
        x = F.relu((self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu((self.dense1(x)))
        x = F.relu((self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(nn.Module):
    
    def __init__(self, k = 128, feature_transform=False):
        super(PointNet, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv2c = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.m   = nn.Dropout(p=0.3)
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv2c(x)))
        x = self.m(x)
        x = self.conv3(x)
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x

