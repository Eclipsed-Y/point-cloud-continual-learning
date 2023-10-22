import copy

import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType
# from utils import utils
import utils.optim
import numpy as np
from geotorch import orthogonal
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
# import pymanopt.solvers
import time
import os, sys
from utils import wasserstein_fusion
from geodesic_algorithms.GFK_distill_normalise import GFK
from utils import orthogonal_loss

gfk = GFK()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


setup_seed(2019)

cur_points = torch.empty(0)
ref_points = torch.empty(0)

cur_features = torch.empty(0)
cur_num = 0
ref_features = torch.empty(0)
ref_num = 0
point_dict = {'table': 0, 'chair': 1, 'airplane': 2, 'car': 3, 'sofa': 4, 'lamp': 5, 'vessel': 6, 'cabinet': 7,
              'monitor': 8, 'guitar': 9, 'bookshelf': 10, 'laptop': 11, 'pistol': 12, 'bed': 13, 'rocket': 14,
              'bag': 15, 'earphone': 16, 'cap': 17}


def get_ref_feature(self, inputs, outputs):
    global ref_features, ref_num
    ref_features = inputs[0]
    # ref_features = outputs
    # ref_features = F.normalize(outputs, dim = 1)
    # print(len(inputs))
    # ref_num += inputs.shape[0]

def get_cur_feature(self, inputs, outputs):
    global cur_features, cur_num
    cur_features = inputs[0]

def get_ref_points(self, inputs, outputs):
    global ref_points
    ref_points = outputs

def get_cur_points(self, inputs, outputs):
    global cur_points
    cur_points = outputs


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MLP_correlation(pl.LightningModule):
    def __init__(self, args, outdim):
        super().__init__()
        self.args = args
        self.model = self.create_model(outdim=outdim)
        self.criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.episodic_mem_size = args.mem_size
        self.mem_data = torch.zeros(self.episodic_mem_size, 2048, 3).to(device)
        self.mem_label = torch.zeros(self.episodic_mem_size).long().to(device)
        self.mem_used = 0
        self.data_seen = 0
        self.eps_mem_batch = args.mem_batch
        self.diameter = list()
        # self.model.summary()
        # print(self.model.last.weight.shape)

    def create_model(self, outdim):
        model = models.mlp.PointNet(outdim)
        # model.weights_init()
        return model

    def forward(self, x):
        return self.model.forward(x)

    def weight_init(self):
        for param in self.parameters():
            if len(param.shape) == 2:
                param /= param.norm(dim=1, keepdim=True)
            elif len(param.shape) == 1:
                param /= param.norm()

    def configure_optimizers(self):
        # print(self.named_parameters())
        # 标准
        return torch.optim.Adam(self.parameters(), lr=0.001)
        # return torch.optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9)
        # return utils.optim.ObliqueOpt(self.parameters(), lr = 0.1, momentum = 0.9)
        # return utils.optim.ObliqueOpt(self.parameters(), lr = 0.01)
        # return torch.optim.Adam(self.parameters(), lr = 1e-3)
        # 乘法实验
        # return torch.optim.SGD(self.parameters(), lr = 0.05 / (self.args.multiple ** 2))

        # 除法实验
        # return torch.optim.SGD(self.parameters(), lr = 0.1 * self.args.multiple)

        # return torch.optim.SGD(self.parameters(), lr = 0.003)
        # return torch.optim.Adadelta(self.parameters(),lr= 0.3)
        # return torch.optim.Adam(self.parameters(), lr = 0.1 / self.args.multiple)

    def training_step(self, batch, batch_idx):
        # print(torch.diag(self.model.linear[0].weight @ self.model.linear[0].weight.T))
        x, y, task = batch
        unique_task = torch.unique(task)
        loss = 0
        for i in range(unique_task.shape[0]):
            ttask = unique_task[i]
            tx = x[task == ttask]
            ty = y[task == ttask]
            tt = task[task == ttask]
            y_hat = self.model(tx)
            loss += self.criterion(y_hat, ty)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, task = batch
        unique_task = torch.unique(task)
        cnt, acc, loss = 0, 0, 0
        for i in range(unique_task.shape[0]):
            ttask = unique_task[i]
            tx = x[task == ttask]
            ty = y[task == ttask]
            tt = task[task == ttask]
            y_hat = self.model(tx, tt)
            loss += self.criterion(y_hat, ty)
            acc += (y_hat.argmax(dim=1) == ty).float().sum().item()
            cnt += y_hat.shape[0]
        acc = acc / cnt
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def accuracy(self, output, target):
        return (output.argmax(dim=1) == target).float().mean().item()

    def cosine_similarity(self, x, y):
        return torch.dot(x.view(-1), y.view(-1)) / (x.norm() * y.norm())

    def similarity(self, ref_model):
        tot_similarity = 0
        param_dict = dict(ref_model.named_parameters())
        for name, param in self.model.named_parameters():
            # view parameter matrix as a tensor, better performance
            tot_similarity += 1 - self.cosine_similarity(param, param_dict[name])
            # norm_1 = param.norm(dim = 1, keepdim = True)
            # norm_2 = param_dict[name].norm(dim = 1, keepdim = True)
            # tot_similarity += 1 - torch.diag(param @ param_dict[name].T / (norm_1 @ norm_2.T)).mean()
        return tot_similarity

    def class_incremental(self, cl_number):
        '''

        :param cl_number: The number of incremental class
        :return:
        '''
        in_features = self.model.last.in_features
        out_features = self.model.last.out_features
        new_out_features = out_features + cl_number

        # new classifier
        new_last = nn.Linear(in_features, new_out_features)
        nn.init.constant(new_last.weight, 0)
        new_last.weight.data[:out_features, :] = self.model.last.weight.data
        return new_last

    def get_grad_projector(self, ref_model) -> dict:
        # param_dict = dict(ref_model.named_parameters())
        projector_dict = dict()
        for name, p in ref_model.named_parameters():
            if len(p.shape) == 2:
                U, S, Vh = torch.linalg.svd(p.T, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                # r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
                r = torch.sum(torch.cumsum(sval_ratio, 0) < 0.99)  # +1
                projector_dict[name] = U[:, 0:r]
                projector_dict[name] = projector_dict[name] @ projector_dict[name].T
                # feature_list.append(U[:, 0:r])
        return projector_dict

    def curvature(self, points, k):

        # 将点云数据转换为合适的形状
        x = points.permute(0, 2, 1)  # 形状变为 (n, num_points, num_features)

        # 计算距离矩阵
        dist = torch.cdist(x, x)

        # 排除自身距离，然后排序并选取前k个最近邻距离
        sorted_distances, _ = dist.sort(dim=-1)
        distances = sorted_distances[:, :, 1:k+1]  # 排除自身，保留最近的k个距离

        # 计算曲率信息
        curvature_info = 1.0 / (1e-8 + distances.mean(dim=-1, keepdim=True))

        return curvature_info

    def train(self, train_loader, val_loader, task_num=0, ref_model=None):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        if task_num > 0:  # 冻结之前的head
            pre_out = (task_num - 1) * 2 + 4
            self.model.freeze_layer(self.model.output_layer[0: pre_out])

        self.model.train()  # 将model设置为训练模式
        opt = self.configure_optimizers()  # 设置为config里的优化器，默认为SGD（随机梯度下降）
        # print('<================== Task {} ===================>'.format(task_num))

        # if ref_model is not None:
        #     projector = self.get_grad_projector(ref_model)
        if ref_model is not None:
            ref_model.eval()
            ref_hook = ref_model.output_layer[0].register_forward_hook(get_ref_feature)
            cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)
            ref_points_hook = ref_model.feature_layer[8].register_forward_hook(get_ref_points)
            cur_points_hook = self.model.feature_layer[8].register_forward_hook(get_cur_points)


        best_Acc = 0

        for e in range(5):
            tot_loss = 0
            tot_size = 0
            end = time.time()
            for batch in train_loader:
                x = batch['pointcloud'].to(device)
                y_tmp = batch['cate']
                y = torch.zeros(len(y_tmp)).long()
                for i in range(len(y_tmp)):
                    y[i] = point_dict[y_tmp[i]]
                y = y.to(device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)  # 计算损失函数，默认为CrossEntropyLoss
                if ref_model is not None:
                    ref_model = ref_model.to(device)
                    mm_loss = self.similarity(ref_model)  # 计算余弦相似度？
                    loss += mm_loss
                    # fusing feature manifold in replay
                    er_mem_indices = np.random.choice(self.mem_used, min(self.mem_used, self.eps_mem_batch),
                                                      replace=False)  # 生成一个以eps_mem_batch（默认为16）大小的一维数组，其中数值为0-mem_used的随机数
                    er_mem_indices = torch.from_numpy(er_mem_indices).to(x.device).long()
                    old_x = self.mem_data[er_mem_indices]
                    old_y = self.mem_label[er_mem_indices]
                    with torch.no_grad():
                        old_y_ref_hat = ref_model(old_x)
                    old_y_hat = self.model(old_x)

                    # 关系蒸馏
                    # cur_feature_result = torch.matmul(cur_features, cur_features.T)
                    # ref_feature_result = torch.matmul(ref_features, ref_features.T)
                    # feature_mse_loss = F.mse_loss(cur_feature_result, ref_feature_result) / 1600000
                    # loss += feature_mse_loss

                    # 曲率蒸馏
                    cur_curve = self.curvature(cur_points, 5).squeeze(-1)  # 移除最后一维尺寸为1的部分
                    ref_curve = self.curvature(ref_points, 5).squeeze(-1)
                    curvature_loss = self.feature_matching_loss(cur_curve, ref_curve)
                    loss += 10 * curvature_loss


                    feature_loss = self.feature_matching_loss(cur_features, ref_features)  # 原始的蒸馏方式  16*256
                    loss += feature_loss
                    # lamda = 10
                    # loss += gfk.fit(ref_features.detach(), cur_features) * lamda  # grassmann方式

                    replay_loss = self.criterion(old_y_hat, old_y)  # 计算旧模型预测的loss
                    loss += replay_loss

                    margin_loss = orthogonal_loss.margin_loss(self.model.output_layer[0: pre_out], self.model.output_layer[pre_out: pre_out + 2])
                    loss += margin_loss  # 添加margin损失

                    # print(f"mm_loss = {mm_loss}, feature_matching_loss = {feature_loss}, replay_loss = {replay_loss}, margin_loss = {margin_loss}")

                opt.zero_grad()
                loss.backward()
                tot_loss += loss.item()
                tot_size += 1
                opt.step()
                # if ref_model is not None:
                #     print(f'similarity: {self.similarity(ref_model)}')

            Acc = self.validation(val_loader)  # 计算当前轮次验证集的准确率（从开始到现在所有类别）

            time_cost = time.time() - end
            if ref_model is not None:
                print(f'<=== epoch: {e}    loss: {tot_loss / tot_size}   acc: {Acc}   time: {time_cost}  similarity: {self.similarity(ref_model)}===>')
                # print(f'<=== epoch: {e}    loss: {tot_loss / tot_size}   time: {time_cost}  matching_loss: {self.manifold_matching_loss(ref_model)} ===>')
            else:
                print(f'<=== epoch: {e}    loss: {tot_loss / tot_size}    acc: {Acc}   time: {time_cost} ===>')
            if Acc > best_Acc:
                torch.save(self.model.state_dict(), r'checkpoint/pointcloud/PointBest.pkl')
                best_Acc = Acc
        # Model fusion
        # print(f'Acc: {self.validation(val_loader)}')
        # if ref_model is not None:
        #     self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[self.model, ref_model])
        # self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[ref_model, self.model])
        #
        # if ref_model is not None:
        #     param_dict = dict(ref_model.named_parameters())
        #     for name, p in self.model.named_parameters():
        #         p.data = 0.2 * param_dict[name].data + 0.8 * p.data

        self.model.load_state_dict(torch.load(r'checkpoint/pointcloud/PointBest.pkl'))  # 选取准确率最好的模型作为当前模型
        self.update_memory(train_loader)
        self.calc_diameter()
        if (ref_model is not None):
            ref_hook.remove()
            cur_hook.remove()
            ref_points_hook.remove()
            cur_points_hook.remove()
        return self.validation(val_loader)

    def validation(self, val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        cnt, acc = 0, 0
        for batch in val_loader:
            x = batch['pointcloud'].to(device)
            y_tmp = batch['cate']
            y = torch.zeros(len(y_tmp)).long()
            for i in range(len(y_tmp)):
                y[i] = point_dict[y_tmp[i]]
            y = y.to(device)
            with torch.no_grad():
                y_hat = self.model(x)
                acc += (y_hat.argmax(dim=1) == y).float().sum().item()
                cnt += y_hat.shape[0]
        return acc / cnt

    def pairwise_distances(self, x, y):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        # x = x / x.norm(dim = 1)
        # y = y / y.norm(dim = 1)
        # x = F.normalize(x, dim = 1)
        # y = F.normalize(y, dim = 1)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def manifold_matching_loss(self, ref_model):
        tot_similarity = 0
        tot_manifold_matching = 0
        param_dict = dict(ref_model.named_parameters())
        for name, param in self.model.named_parameters():
            tot_manifold_matching += self.geodesic_dist(param, param_dict[name])

            # if len(param.shape) == 2:
            #     dis_s = self.pairwise_distances(param, param)
            #     dis_t = self.pairwise_distances(param_dict[name], param_dict[name])
            #     tot_manifold_matching += torch.dist(dis_s,dis_t,2) + torch.dist(param, param_dict[name], 2)

        return tot_manifold_matching

    def geodesic_dist(self, X, Y):
        # return (X - Y).norm()
        XY = (X.data * Y.data).sum(0)
        XY[XY > 1] = 1
        U = torch.arccos(XY)
        return torch.norm(U)

    def feature_matching_loss(self, X, Y, eps=1e-9):
        # X += eps
        # Y += eps
        # X = X / X.norm(dim = 1)
        # Y = Y / Y.norm(dim = 1)
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        # geodesic = self.Sphere_dist(X,Y).trace() / X.shape[0]
        # cost = X @ Y.T
        # geodesic = (cost - 2 * torch.diag(cost) + torch.eye(cost.shape[0]).to(cost.device)).mean()
        # geodesic = (-1 * torch.diag(cost) + torch.eye(cost.shape[0]).to(cost.device)).mean()
        # geodesic
        # geodesic = -1 * torch.eye(X.shape[0]) - torch.diag(X @ Y.T).mean()
        geodesic = torch.diag(self.pairwise_distances(X, Y)).mean()
        # geodesic = self.pairwise_distances(X,Y).mean()
        return geodesic

    def calc_diameter(self):
        cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)
        self.model(self.mem_data)
        features = F.normalize(cur_features, dim=1)
        center = features.mean(dim=0, keepdim=True)
        distance = self.pairwise_distances(center, features).squeeze()
        sorted_index = torch.argsort(distance)
        self.diameter.append(distance[sorted_index[-1]].item())
        cur_hook.remove()

    def update_memory(self, dataset_loader):
        self.model.eval()

        cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)  # 钩子函数
        replay_feature, center, distance, sorted_index = None, None, None, None
        for batch in dataset_loader:
            x_batch = batch['pointcloud']
            y_tmp = batch['cate']
            y_batch = torch.zeros(len(y_tmp)).long()
            for i in range(len(y_tmp)):
                y_batch[i] = point_dict[y_tmp[i]]
            for index in range(x_batch.shape[0]):
                x, y = x_batch[index], y_batch[index]
                if self.episodic_mem_size > self.mem_used:  # 如果当前memory库存没有到上限，那就直接把x和y丢进库存里
                    self.mem_data[self.mem_used] = x
                    self.mem_label[self.mem_used] = y
                    self.mem_used += 1
                else:
                    if center == None:
                        self.model(self.mem_data)
                        replay_feature = F.normalize(cur_features, dim=1)
                        center = replay_feature.mean(dim=0, keepdim=True)
                        distance = self.pairwise_distances(center, replay_feature).squeeze()
                        sorted_index = torch.argsort(distance)
                    with torch.no_grad():
                        self.model(torch.reshape(x, shape=(-1, 2048, 3)).cuda())
                    dis_to_center = torch.dist(F.normalize(cur_features, dim=1), center, 2) ** 2
                    if dis_to_center > distance[sorted_index[-1]]:
                        j = np.random.randint(0, self.mem_used)
                        self.mem_data[j], self.mem_label[j] = x, y
                        distance[j] = dis_to_center
                        sorted_index = torch.argsort(distance)
                    # elif dis_to_center >= distance[sorted_index[0]]:
                    else:
                        j = np.random.randint(0, self.data_seen)
                        # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
                        if j < self.episodic_mem_size:
                            self.mem_data[sorted_index[j]], self.mem_label[sorted_index[j]] = x, y
                            distance[sorted_index[j]] = dis_to_center
                            sorted_index = torch.argsort(distance)

                    # j = np.random.randint(0, self.data_seen)
                    # # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
                    # if j < self.episodic_mem_size:
                    #     self.mem_data[j], self.mem_label[j] = x, y
                self.data_seen += 1
        cur_hook.remove()

    def update_reservior(self, current_image, current_label):
        """
        Update the episodic memory with current example using the reservior sampling
        """
        if self.episodic_mem_size > self.data_seen:
            self.mem_data[self.data_seen] = current_image
            self.mem_label[self.data_seen] = current_label
            # self.mem_task_id[self.data_seen] = current_task
            self.mem_used += 1
        else:
            j = np.random.randint(0, self.data_seen)
            if j < self.episodic_mem_size:
                self.mem_data[j] = current_image
                self.mem_label[j] = current_label
                # self.mem_task_id[j] = current_task

        self.data_seen += 1

    # def update_reservior(self, current_image, current_label):
    #

    def Sphere_dist(self, x, y, eps=1e-9):
        # Make sure inner product is between -1 and 1
        # assert not torch.isnan(x) and not torch.isnan(y)
        inner = x @ y.T
        # torch.clamp_(inner, -1 + eps, 1 - eps)
        inner = torch.clamp(inner, min=-1, max=1)
        # inner = max(min(inner, 1), -1)
        # print(inner.max(),inner.min())
        assert inner.max() <= 1 and inner.min() >= -1
        return torch.arccos(inner)
