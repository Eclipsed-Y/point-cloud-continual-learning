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
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=8,num_channels=planes)
        self.conv1 = nn.Parameter(torch.rand(planes,in_planes,3,3))
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=8, num_channels=planes)
        self.conv2 = nn.Parameter(torch.rand(planes,planes,3,3))
        self.planes = planes
        self.in_planes = in_planes
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_conv = nn.Parameter(torch.rand(self.expansion * planes,in_planes,1,1))
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion * planes)
            # )
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
        # for m in self.modules():
        #     if isinstance(m,nn.BatchNorm2d):
        #         nn.init.constant_(m.weight,1)
        #         nn.init.constant_(m.bias,0)
    def conv_proj(self,kernel,proj):
        shape = kernel.shape
        new_kernel = (kernel.view(-1,shape[0]) @ proj).view(shape)
        # (out,int,k,k)
        # print(new_kernel.max())
        return new_kernel
    def feature_proj(self,x,proj):
        # (batch, out, k, k)
        new_feature = (x.transpose(1,3) @ proj).transpose(1,3)
        return new_feature
    def forward(self, x,task):
        # (batch_size, C, H, W)
        key = str(int(task[0]))
        # out = F.relu(self.feature_proj(self.bn1(x),self.proj[key]))
        out = x
        shortcut = x
        if hasattr(self,'shortcut_conv'):
            # shortcut = self.shortcut_bn(torch.conv2d(out,self.conv_proj(self.shortcut_conv,self.proj[key]),stride=self.stride,padding= 'same' if self.stride == 1 else 'valid'))
            shortcut = torch.conv2d(out, self.shortcut_conv, stride=self.stride,padding='same' if self.stride == 1 else 'valid')
            shortcut = self.feature_proj(shortcut,self.proj[key])
            # shortcut = self.shortcut_bn(shortcut)
            shortcut = self.shortcut_bn(shortcut)
            # shortcut = self.feature_proj(shortcut,self.proj[key])

        # out = self.bn1(torch.conv2d(out,self.conv_proj(self.conv1,self.proj[key]),stride=self.stride,padding = 1))
        out = torch.conv2d(out,self.conv1, stride=self.stride, padding = 1)
        out = self.feature_proj(out,self.proj[key])
        # out = self.bn1(out)
        out = self.bn1(out)
        # out = self.feature_proj(out,self.proj[key])
        out = F.relu(out)
        # print(out.shape)

        out = torch.conv2d(out,self.conv2,stride=1,padding = 1)
        out = self.feature_proj(out, self.proj[key])
        # out = self.bn2(out)
        out = self.bn2(out)
        # out = self.feature_proj(out,self.proj[key])
        # print(out.shape,shortcut.shape)
        out += shortcut
        out = F.relu(out)
        return out

    # def forward(self, x,task):
    #     # (batch_size, C, H, W)
    #     task = (task - 1) * x.shape[0] + torch.Tensor(range(x.shape[0])).long().to(x.device)
    #     out = F.relu(self.bn1(x))
    #     shortcut = x
    #     if hasattr(self,'shortcut_conv'):
    #         shortcut = torch.cat([self.shortcut_bn(torch.conv2d(out,self.conv_proj(self.shortcut_conv,proj))) for task,proj in self.proj.items()],0)[task]
    #
    #     out = self.conv1(out)
    #     out = torch.cat([(out.transpose(1,3) @ self.proj[task]).transpose(1,3)
    #                    for task in self.proj.keys()],0)[task]
    #     out = self.conv2(F.relu(self.bn2(out)))
    #     # 投影
    #     out = torch.cat([(out.transpose(1,3) @ self.proj[task]).transpose(1,3)
    #                    for task in self.proj.keys()],0)[task]
    #     out += shortcut
    #     return out

class PreActResNet_PROJ(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(PreActResNet_PROJ, self).__init__()
        self.in_planes = 64
        last_planes = 512*block.expansion
        self.last_plane = last_planes
        # self.conv1 = conv3x3(in_channels, 64)

        self.conv1 = nn.Parameter(torch.rand(64,in_channels,7,7))
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.GroupNorm(num_groups=8,num_channels=64)
        self.proj = nn.ParameterDict()
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

        self.weights_init()
    def conv_proj(self,kernel,proj):
        shape = kernel.shape
        new_kernel = (kernel.view(shape[1],-1) @ proj).view(shape)
        # (out,int,k,k)
        return new_kernel

    def feature_proj(self,x,proj):
        # (batch, out, k, k)
        new_feature = (x.transpose(1,3) @ proj).transpose(1,3)
        return new_feature

    def weights_init(self):
        nn.init.xavier_uniform(self.conv1)
        nn.init.xavier_uniform(self.last.weight)
        nn.init.constant_(self.last.bias,0.1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def features(self, x,task):
        key = str(int(task[0]))
        # out = self.bn1(torch.conv2d(x,self.conv_proj(self.conv1,self.proj[key]),stride = 1, padding = 1))

        out = torch.conv2d(x,self.conv1,stride = 1, padding = 1)
        # out = torch.conv2d(x,self.conv_proj(self.conv1,self.proj[key]),stride = 1, padding = 1)
        out = self.feature_proj(out, self.proj[key])
        out = F.relu(self.bn1(out))
        # out = self.feature_proj(out,self.proj[key])
        out = self.stage1[0](out, task)
        out = self.stage1[1](out, task)
        out = self.stage2[0](out, task)
        out = self.stage2[1](out, task)
        out = self.stage3[0](out, task)
        out = self.stage3[1](out, task)
        out = self.stage4[0](out, task)
        out = self.stage4[1](out, task)
        return out

    def logits(self, x, task):
        x = self.last(x)
        return x

    def forward(self, x, task):
        x = self.features(x, task)
        # x = F.relu(self.bn_last(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.logits(x.view(x.size(0), -1), task)
        return x


    def deconv_orth_dist(self,kernel, stride=2, padding=1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c).cuda()
        return torch.norm(output - target)

    def orth_dist(self,mat, stride=None):
        mat = mat.reshape((mat.shape[0], -1))
        if mat.shape[0] < mat.shape[1]:
            mat = mat.permute(1, 0)
        return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).cuda())

    def OrthogonalRegular(self):
        diff = self.orth_dist(self.stage2[0].shortcut_conv) + self.orth_dist(self.stage3[0].shortcut_conv) + self.orth_dist(self.stage4[0].shortcut_conv)
        diff += self.deconv_orth_dist(self.stage1[0].conv1, stride=1) + self.deconv_orth_dist(self.stage1[1].conv1, stride=1)
        diff += self.deconv_orth_dist(self.stage2[0].conv1, stride=2) + self.deconv_orth_dist(self.stage2[1].conv1, stride=1)
        diff += self.deconv_orth_dist(self.stage3[0].conv1, stride=2) + self.deconv_orth_dist(self.stage3[1].conv1, stride=1)
        diff += self.deconv_orth_dist(self.stage4[0].conv1, stride=2) + self.deconv_orth_dist(self.stage4[1].conv1, stride=1)
        return diff


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        last_planes = 512*block.expansion
        self.last_plane = last_planes
        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

        # Only for OGD+ purpose, don't pay attention to the name
        self.linear = nn.Sequential(
            self.conv1,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x,task):
        out = self.conv1(x)
        out = self.stage1[0](out, task)
        out = self.stage1[1](out, task)
        out = self.stage2[0](out, task)
        out = self.stage2[1](out, task)
        out = self.stage3[0](out, task)
        out = self.stage3[1](out, task)
        out = self.stage4[0](out, task)
        out = self.stage4[1](out, task)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, task):
        x = self.features(x, task)
        x = F.relu(self.bn_last(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.logits(x.view(x.size(0), -1))
        return x
    def weights_init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                indim = m.weight.shape[1]
                nn.init.normal_(m.weight,0,1.0 / np.sqrt(indim))
                # nn.init.xavier_uniformd(m.weight.data)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def OrthogonalConstraint(self):
        for module in list(self.modules()):
            if hasattr(module, 'weight'):
                if len(module.weight.shape) >= 2:
                    if not hasattr(module,'out_features') or module.out_features != self.last_plane:
                        orthogonal(module, "weight", triv='cayley')


class PreActResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, droprate=0):
        super(PreActResNet_cifar, self).__init__()
        self.in_planes = 16
        last_planes = filters[2]*block.expansion

        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1, droprate=droprate)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2, droprate=droprate)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2, droprate=droprate)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

        # Only for OGD+ purpose, don't pay attention to the name
        self.linear = nn.Sequential(
            self.conv1,
            self.stage1,
            self.stage2,
            self.stage3,
        )

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        """

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, droprate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.features(x)
        out = F.relu(self.bn_last(out))
        out = F.avg_pool2d(out, 8)
        out = self.logits(out.view(out.size(0), -1))
        return out


# ResNet for Cifar10/100 or the dataset with image size 32x32

def ResNet20_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def ResNet56_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [9 , 9 , 9 ], [16, 32, 64], num_classes=out_dim)

def ResNet110_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def ResNet29_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBottleneck, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

# def ResNet164_cifar(out_dim=10):
#     return PreActResNet_cifar(PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def ResNet164_cifar(out_dim=5):
    return PreActResNet_cifar(PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes=out_dim)


def WideResNet_28_2_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def WideResNet_28_2_drop_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim, droprate=0.3)

def WideResNet_28_10_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [160, 320, 640], num_classes=out_dim)

# ResNet for general purpose. Ex:ImageNet

def ResNet10(out_dim=10):
    return PreActResNet(PreActBlockProj, [1,1,1,1], num_classes=out_dim)

def ResNet18S(out_dim=10):
    return PreActResNet(PreActBlockProj, [2,2,2,2], num_classes=out_dim, in_channels=1)

def ResNet18(out_dim=10):
    return PreActResNet(PreActBlockProj, [2,2,2,2], num_classes=out_dim)

def ResNet18_PROJ(out_dim = 10):
    return PreActResNet_PROJ(PreActBlock_PROJ, [2, 2, 2, 2], num_classes=out_dim)
def ResNet34(out_dim=10):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes=out_dim)

def ResNet50(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=out_dim)

def ResNet101(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes=out_dim)

def ResNet152(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=out_dim)