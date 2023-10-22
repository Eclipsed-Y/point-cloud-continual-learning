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
import time

cur_features = torch.empty(0)
cur_num = 0
ref_features = torch.empty(0)
ref_num = 0

def get_ref_feature(self, inputs, outputs):
    global ref_features, ref_num
    ref_features = inputs[0]
    # ref_features = F.normalize(outputs, dim = 1)
    # print(len(inputs))
    # ref_num += inputs.shape[0]

def get_cur_feature(self, inputs, outputs):
    global cur_features, cur_num
    cur_features = inputs[0]
    # cur_features = F.normalize(outputs, dim = 1)
    # cur_num += inputs.shape[0]

class MLP_Manifold_Matching(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = self.create_model(args)
        self.criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.mem_data = torch.zeros(self.episodic_mem_size, 3, 32, 32).to(device)
        # self.mem_label = torch.zeros(self.episodic_mem_size).long().to(device)
        # self.centroids = dict()

    def create_model(self,args = None):
        model = models.mlp.MLP256()
        # model.weights_init()
        return model

    def forward(self, x):
        return self.model.forward(x)
    def weight_init(self):
        for param in self.parameters():
            if len(param.shape) == 2:
                param /= param.norm(dim = 1,keepdim = True)
            elif len(param.shape) == 1:
                param /= param.norm()


    def configure_optimizers(self):
        # print(self.named_parameters())
        # 标准
        # return torch.optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9)
        return torch.optim.Adam(self.parameters(), lr = 1e-3)
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
            y_hat = self(tx)
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
            y_hat = self(tx, tt)
            loss += self.criterion(y_hat, ty)
            acc += (y_hat.argmax(dim=1) == ty).float().sum().item()
            cnt += y_hat.shape[0]
        acc = acc / cnt
        self.log('val_loss',loss)
        self.log('val_acc',acc)

    def accuracy(self,output,target):
        return (output.argmax(dim = 1) == target).float().mean().item()

    def cosine_similarity(self, x, y):
        return torch.dot(x.view(-1), y.view(-1)) / (x.norm() * y.norm())
        # if len(x.shape) == 2:
        #     return (x.T @ y).trace() / (x.norm() * y.norm())
        # else:
        #     return torch.dot(x,y) / (x.norm() * y.norm())
        # return torch.dot(x,y) / (x.norm() * y.norm())

    def similarity(self, ref_model):
        tot_similarity = 0
        param_dict = dict(ref_model.named_parameters())
        # print(param_dict.keys())
        for name, param in self.model.named_parameters():
            # tot_similarity += self.cosine_similarity(param.data.view(-1),param_dict[name].data.view(-1))
            tot_similarity += 1 - self.cosine_similarity(param, param_dict[name])
            # tot_similarity += torch.dot(param.data.view(-1), param_dict[name].data.view(-1)) / (param.data.norm() * param_dict[name].data.norm())
        return tot_similarity

    def class_incremental(self,cl_number):
        '''

        :param cl_number: The number of incremental class
        :return:
        '''
        in_features = self.model.last.in_features
        out_features = self.model.last.out_features
        new_out_features = out_features + cl_number

        # new classifier
        new_last = nn.Linear(in_features,new_out_features)
        new_last.weight.data[:out_features] = self.model.last.weight.data
        new_last.bias.data[:out_features] = self.model.last.bias.data
        self.model.last = new_last

    def train(self, train_loader, val_loader, task_num = 0, ref_model = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.train()
        opt = self.configure_optimizers()
        if ref_model is not None:
            ref_model.eval()
            ref_hook = ref_model.last.register_forward_hook(get_ref_feature)
            cur_hook = self.model.last.register_forward_hook(get_cur_feature)
        # print('<================== Task {} ===================>'.format(task_num))
        for e in range(10):
            tot_loss = 0
            tot_matching = 0
            tot_size = 0
            end = time.time()
            for x, y, _ in train_loader:
                x = x.to(device)
                y = y.to(device)
                y_cur_hat = self(x)
                loss = self.criterion(y_cur_hat,y)

                if ref_model is not None:
                    with torch.no_grad():
                        y_ref_hat = ref_model(x)
                    # matching_loss = 0.01 * self.feature_matching_loss(cur_features, ref_features) + self.similarity(ref_model)
                    # matching_loss = self.feature_matching_loss(cur_features, ref_features) + 10 * self.similarity(ref_model)
                    matching_loss = self.feature_matching_loss(cur_features, ref_features)
                    # matching_loss = 10 * self.similarity(ref_model)

                    # cosine similarity
                    # cur_features = F.normalize(cur_features, dim = 1)
                    # ref_features = F.normalize(ref_features, dim = 1)
                    # matching_loss = (1 - (cur_features @ ref_features.T)).trace()
                    loss += matching_loss

                opt.zero_grad()
                loss.backward()
                tot_loss += loss.item()
                if ref_model is not None:
                    tot_matching += matching_loss.item()
                tot_size += x.shape[0]
                opt.step()
            time_cost = time.time() - end
            print(f'<=== epoch: {e}    loss: {tot_loss / tot_size} matching_loss : {tot_matching / tot_size}   time: {time_cost} ===>')

        if (ref_model is not None):
            ref_hook.remove()
            cur_hook.remove()

        return self.validation(val_loader)

    def validation(self,val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        cnt, acc = 0, 0
        for x, y, task in val_loader:
            x = x.to(device)
            y = y.to(device)
            task = task.to(device)
            unique_task = torch.unique(task)
            for i in range(unique_task.shape[0]):
                with torch.no_grad():
                    ttask = unique_task[i]
                    tx = x[task == ttask]
                    ty = y[task == ttask]
                    tt = task[task == ttask]
                    y_hat = self(tx)
                    acc += (y_hat.argmax(dim=1) == ty).float().sum().item()
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
        x = F.normalize(x, dim = 1)
        y = F.normalize(y, dim = 1)
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

    def feature_matching_loss(self, X, Y):
        X_center, Y_center = X.mean(0), Y.mean(0)
        center_dist = torch.dist(X_center,Y_center,2)
        X_dist = self.pairwise_distances(X, X)
        Y_dist = self.pairwise_distances(Y, Y)
        p_dist = torch.dist(X_dist, Y_dist, 2)
        # return center_dist
        # return X_dist.sum()
        # return p_dist
        return center_dist + p_dist





