import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k 
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        n_points = x.size(2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(kernel_size=n_points)(x)
        x = nn.Flatten(start_dim=1)(x) 
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        init = torch.eye(self.k, requires_grad=True).repeat(batchsize,1,1) 
        if x.is_cuda: 
            init = init.cuda()
        matrix = self.fc3(x).view(-1,self.k,self.k) + init 
        return matrix

    
class Transform(nn.Module):
    def __init__(self, global_feature=True):
        super(Transform, self).__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feature = global_feature 

    def forward(self, x):
        n_points = x.size(2)

        m3x3 = self.input_transform(x)
        x = torch.bmm(torch.transpose(x,1,2), m3x3).transpose(1,2) # batch matrix multiplication
        x = F.relu(self.bn1(self.conv1(x)))

        m64x64 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x,1,2), m64x64).transpose(1,2) # batch matrix multiplication
        localfeat = x # extract local features for segmentation task 

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(kernel_size=x.size(-1))(x)
        out = nn.Flatten(start_dim=1)(x)
        
        if self.global_feature: 
            return out, m3x3, m64x64
        else: 
            out = out.view(-1,1024,1).repeat(1,1,n_points)
            return torch.cat([localfeat, out], dim=1), m3x3, m64x64


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.transform = Transform(global_feature=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, m3x3, m64x64 = self.transform(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(1,2)
        return x, m3x3, m64x64
