import torch
import torch.nn as nn
# from geotorch import Stiefel, orthogonal, grassmannian
# import geotorch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

setup_seed(2019)

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class MLP(nn.Module):
    def __init__(self, outdim, hidden_dim=256, dropout=0.3):
        super(MLP, self).__init__()
        self.in_dim = 2048 * 3
        self.hidden_dim = hidden_dim
        self.out_dim = outdim
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim, bias = False),
            # nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias = False),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, outdim, bias = False)  # Subject to be replaced dependent on task
        # self.last = CosineLinear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        # self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        # self.weights_init()

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    def weights_init(self):
        for module in list(self.modules()):
            if hasattr(module, 'weight'):
                nn.init.normal(module.weight, mean = 0.0, std = 1.0)
                # self.orthogonal_init(module)

    def orthogonal_init(self,layer):
        nn.init.orthogonal(layer.weight)
        if hasattr(layer,'bias'):
            nn.init.constant_(layer.bias,0.1)

    def change_out_dim(self, outdim):
        self.last = nn.Linear(self.hidden_dim, outdim, bias=False)
        self.out_dim = outdim

    # def OrthogonalConstraint(self):
    #     for module in list(self.modules()):
    #         if hasattr(module, 'weight'):
    #             if len(module.weight.shape) >= 2:
    #                 if module.out_features != self.out_dim:
    #                     orthogonal(module, "weight", triv='cayley')

class PointNet(nn.Module):
    def __init__(self, outdim):
        super(PointNet, self).__init__()

        # 特征提取层
        self.out_dim = 0
        self.feature_layer = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 共享隐藏层
        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)


    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x的形状：(batch_size, 3, num_points)
        x = self.feature_layer(x)
        # x的形状：(batch_size, 1024, num_points)
        x, _ = torch.max(x, 2)
        # x的形状：(batch_size, 1024)
        x = self.shared_layer(x)
        # x的形状：(batch_size, 256)
        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)
        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False

class PointNetCurvature(nn.Module):
    def __init__(self, outdim):
        super(PointNetCurvature, self).__init__()

        # 特征提取层
        self.out_dim = 0
        self.feature_layer1 = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.feature_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 共享隐藏层
        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)

    # def curvature(self, points, k):
    #     curvature_info = torch.zeros_like(points[:, :, 0:1])
        # for i in range(points.shape[0]):
        #     print(f"num{i} start!")
        #     x = points[i]  # 2048 * 特征数
        #     dist = torch.cdist(x, x)
        #     distances, indices = dist.topk(k + 1, largest=False)  # topk函数获得k个最近邻点（包括自身），largest=False表示获取最小的距离
        #     for j in range(x.shape[0]):
        #         tmp = 0
        #         for e in range(k):
        #             node = indices[j][e + 1]
        #             for g in range(k):
        #                 if indices[node][g + 1] != j:
        #                     tmp += math.sqrt(distances[node][g + 1])
        #             for g in range(k):
        #                 if indices[j][g + 1] != node:
        #                     tmp += math.sqrt(distances[j][g + 1] / distances[j][e + 1])
        #         curvature_info[i][j] = (2 - tmp) / k
        # points = torch.cat((points, curvature_info), dim=2)
        # return points

    def curvature(self, points, k):
        # start = time.time()
        n, num_points, num_features = points.shape
        curvature_info = torch.zeros(n, num_points, 1, device=points.device)

        for i in range(n):
            x = points[i]  # 2048 * 特征数
            dist = torch.cdist(x, x)
            distances, _ = dist.topk(k + 1, largest=False)  # topk函数获得k个最近邻点（包括自身），largest=False表示获取最小的距离
            distances = distances[:, 1:]
            curvature_info[i] = 1.0 / (1e-8 + distances.mean(dim=1, keepdim=True))

        points = torch.cat((points, curvature_info), dim=2)
        # end = time.time()
        # print(f"cost_time = {end - start}s")
        return points

    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1


    def forward(self, x):
        x = self.curvature(x, 5)
        x = x.permute(0, 2, 1)
        # x的形状：(batch_size, 4, num_points)
        x = self.feature_layer1(x)
        x = self.feature_layer2(x)
        # x的形状：(batch_size, 1024, num_points)
        x, _ = torch.max(x, 2)
        # x的形状：(batch_size, 1024)
        x = self.shared_layer(x)
        # x的形状：(batch_size, 256)
        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)
        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False



    #     # 全连接层
    #     self.fc_layers = nn.Sequential(
    #         nn.Linear(1024, 512),
    #         nn.BatchNorm1d(512),
    #         nn.ReLU(),
    #         nn.Linear(512, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Linear(256, outdim),
    #         nn.LogSoftmax(dim=1)
    #     )
    #
    #     #
    #
    # def forward(self, x):
    #     x = x.permute(0, 2, 1)
    #     # x的形状：(batch_size, 3, num_points)
    #     x = self.feature_layer(x)
    #     # x的形状：(batch_size, 1024, num_points)
    #     x, _ = torch.max(x, 2)
    #     # x的形状：(batch_size, 1024)
    #     x = self.fc_layers(x)
    #     # x的形状：(batch_size, 2)
    #     return x
    #
    # def change_out_dim(self, outdim):
    #     in_features = self.fc_layers[-2].in_features
    #     self.fc_layers[-2] = nn.Linear(in_features, outdim)
    #     self.fc_layers[-1] = nn.LogSoftmax(dim=1)
    #     self.out_dim = outdim


class StableMLP(nn.Module):
    # https://github.com/imirzadeh/stable-continual-learning/blob/master/stable_sgd/models.py
    # https://proceedings.neurips.cc/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Supplemental.pdf
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        super(StableMLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class ToyMLP(nn.Module) :
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        self.in_dim = in_channel * img_sz * img_sz
        self.linear = nn.Sequential()
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def MLP50():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=50)


def MLP100():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=100)

def MLP256(outdim):
    print("\n Using MLP256 \n")
    return MLP(outdim=outdim,hidden_dim=256)

def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    print("\n Using MLP1000 \n")
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)

if __name__ == '__main__':
    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    
    model = MLP100()
    n_params = count_parameter(model)
    print(f"LeNetC has {n_params} parameters")
