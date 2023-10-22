import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from geotorch import Stiefel, orthogonal, grassmannian
import torch, numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock_PROJ(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, droprate=0):
        super(PreActBlock_PROJ, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Parameter(torch.rand(planes,in_planes,3,3))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Parameter(torch.rand(planes,planes,3,3))
        self.planes = planes
        self.in_planes = in_planes

        self.proj = nn.ParameterDict()
        self.weights_init()
    def weights_init(self):
        nn.init.xavier_uniform(self.conv1)
        nn.init.xavier_uniform(self.conv2)
        # nn.init.uniform_(self.conv1, 0, np.sqrt(1.0 / self.in_planes))
        # nn.init.uniform_(self.conv2, 0, np.sqrt(1.0 / self.in_planes))
        if hasattr(self,'shortcut_conv'):
            nn.init.xavier_uniform(self.shortcut_conv)
        # nn.init.xavier_uniform()
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def conv_proj(self,kernel,proj):
        shape = kernel.shape
        new_kernel = (kernel.view(-1,shape[0]) * proj).view(shape)
        # (out,int,k,k)
        # print(new_kernel.max())
        return new_kernel

    def forward(self, x,task):
        # (batch_size, C, H, W)
        key = str(int(task[0]))
        out = F.relu(self.bn1(x))
        shortcut = x
        if hasattr(self,'shortcut_conv'):
            shortcut = self.shortcut_bn(torch.conv2d(out,self.conv_proj(self.shortcut_conv,self.proj[key]),stride=self.stride,padding= 'same' if self.stride == 1 else 'valid'))
        out = torch.conv2d(out,self.conv_proj(self.conv1,self.proj[key]),stride=self.stride,padding = 1)
        out = F.relu(self.bn2(out))
        # print(out.shape)
        out = torch.conv2d(out,self.conv_proj(self.conv2,self.proj[key]),stride=1,padding = 1)
        # print(out.shape,shortcut.shape)
        out += shortcut
        return out

class PreActResNet_PROJ(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(PreActResNet_PROJ, self).__init__()
        self.in_planes = 64
        last_planes = 64
        self.last_plane = last_planes
        # self.conv1 = conv3x3(in_channels, 64)
        self.bn1_weight = nn.Parameter(torch.ones(64))
        self.bn1_bias = nn.Parameter(torch.ones(64))

        self.bn2_weight = nn.Parameter(torch.zeros(64))
        self.bn2_bias = nn.Parameter(torch.zeros(64))

        self.proj = nn.ParameterDict()
        self.conv1 = nn.Parameter(torch.rand(64,in_channels,3,3))
        self.bn1 = nn.BatchNorm2d(64,affine=False)
        # self.bn1 = nn.GroupNorm(num_groups=8,num_channels=64)
        self.conv2 = nn.Parameter(torch.rand(64, 64, 3, 3))
        # self.bn2 = nn.BatchNorm2d(64)
        self.bn_last = nn.BatchNorm2d(last_planes,affine=False)
        # self.bn_last = nn.GroupNorm(num_groups=8, num_channels=64)
        self.last = nn.Linear(last_planes, num_classes)

        self.weights_init()
        # print(self.bn1.weight.shape,self.bn1.bias.shape)
    def conv_proj(self,kernel,proj):
        shape = kernel.shape
        new_kernel = (kernel.view(-1,shape[0]) @ proj).view(shape)
        # (out,int,k,k)
        return new_kernel

    def feature_proj(self,feature,proj):
        shape = feature.shape
        new_feature = (feature.transpose(1,3) @ proj).transpose(1,3)
        return new_feature
    def weights_init(self):
        nn.init.xavier_uniform(self.conv1)
        nn.init.xavier_uniform(self.conv2)
        nn.init.constant_(self.bn1_weight, 1)
        nn.init.constant_(self.bn2_weight, 1)
        nn.init.constant_(self.bn1_bias, 0)
        nn.init.constant_(self.bn2_bias, 0)
        # nn.init.uniform_(self.conv1,0,np.sqrt(1.0 / 3))
        # for m in self.modules():
            # if isinstance(m,nn.BatchNorm2d):
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False
                # nn.init.constant_(m.weight,1)
                # nn.init.constant_(m.bias,0)
        nn.init.xavier_uniform(self.last.weight)
        nn.init.constant_(self.last.bias,0.1)

    def features(self, x,task):
        key = str(int(task[0]))
        # out = F.relu(self.bn1(torch.conv2d(x, self.conv1, stride=1, padding=1)))
        # out = F.relu(self.bn_last(torch.conv2d(out, self.conv2, stride=1, padding=1)))
        # out = F.relu(torch.conv2d(x, self.conv1, stride=1, padding=1))
        # out = F.relu(torch.conv2d(out, self.conv2, stride=1, padding=1))
        # return out
        # out = F.relu(torch.conv2d(x, self.conv_proj(self.conv1, self.proj[key]), stride=1, padding=1))

        # out = F.relu(self.bn1(torch.conv2d(x,self.conv_proj(self.conv1,self.proj[key]),stride = 1, padding = 1)))
        out = self.bn1(self.feature_proj(torch.conv2d(x, self.conv1, stride=1, padding=1),self.proj[key]))
        # out = F.relu(self.feature_proj(out,self.proj[key]))
        # out = F.relu(torch.conv2d(out,self.conv_proj(self.conv2,self.proj[key]),stride = 1, padding = 1))
        # out = F.relu(self.bn_last(torch.conv2d(out, self.conv_proj(self.conv2, self.proj[key]), stride=1, padding=1)))
        out = self.bn_last(self.feature_proj(torch.conv2d(out, self.conv2, stride=1, padding=1),self.proj[key]))
        # out = F.relu(self.feature_proj(out, self.proj[key]))
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, task):
        x = self.features(x, task)
        # x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.logits(x.view(x.size(0), -1))
        return x
