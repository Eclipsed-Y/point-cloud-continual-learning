import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType
from utils import utils
import numpy as np
import time
# from torch.optim.lr_scheduler
from geotorch import orthogonal


class AlexNet_MM(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.episodic_mem_size = args.total_classes * args.mem_size_per_class
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mem_data = torch.zeros(self.episodic_mem_size, 3, 32, 32).to(device)
        # used for miniImageNet
        # self.mem_data = torch.zeros(self.episodic_mem_size,3,64,64).to(device)
        self.mem_label = torch.zeros(self.episodic_mem_size).long().to(device)
        self.mem_task_id = torch.zeros(self.episodic_mem_size).long().to(device)

        # self.mem_data = torch.zeros(self.episodic_mem_size, 3, 32, 32)
        # self.mem_data = torch.zeros(self.episodic_mem_size, 3, 64, 64)
        # self.mem_label = torch.zeros(self.episodic_mem_size).long()
        # self.mem_task_id = torch.zeros(self.episodic_mem_size).long()
        self.mem_used = 0
        self.data_seen = 0
        self.eps_mem_batch = args.eps_mem_batch
        self.model = self.create_model(args)
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self, args):
        # model = models.resnet.ResNet18(10)
        model = models.alexnet.AlexNet()
        n_feat = model.last.in_features
        o_last = model.last.out_features
        model.last = nn.ModuleDict()
        for task, out_dim in args.out_dim.items():
            model.last[str(task)] = nn.Linear(n_feat, out_dim, bias=False)

        # model.relu = nn.ReLU()
        def new_logits(self, x):
            outputs = torch.cat([func(x)
                                 for task, func in self.last.items()], 0)
            # (task,batch_size,class) -> (batch_size, task, class)
            return outputs
            # for task, func in self.last.items():
            #     outputs[task] = func(self.relu(x @ self.proj[task].T)).unsqueeze(0)
            #     # outputs[task] = func(x)
            # return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        return model

    def forward(self, x, task):
        out = self.model.forward(x)
        task = ((task - 1) * x.shape[0] + torch.Tensor(range(x.shape[0])).to(x.device)).long()
        task = task.to(x.device)
        return out[task]
        # print(x.device)

    def configure_optimizers(self):
        # print(self.named_parameters())
        # return torch.optim.SGD(self.parameters(), lr = 0.03, momentum = 0.9)
        return torch.optim.SGD(self.parameters(), lr = 0.01, momentum=0.9)
        # return torch.optim.Adam(self.parameters(), lr = 0.001)
        # return torch.optim.SGD(self.parameters(), lr = 0.03)
        # return torch.optim.Adadelta(self.parameters(),lr= 0.3)
        # return torch.optim.Adam(self.parameters(), lr = 0.03)

    def cosine_similarity(self, x, y):
        return torch.dot(x.view(-1), y.view(-1)) / (x.norm() * y.norm())

    def similarity(self, ref_model):
        tot_similarity = 0
        param_dict = dict(ref_model.named_parameters())
        for name, param in self.model.named_parameters():
            # view parameter matrix as a tensor, better performance
            tot_similarity += 1 - self.cosine_similarity(param, param_dict[name])
        return tot_similarity

    def training_step(self, batch, batch_idx):
        x, y, task = batch
        # er_mem_indices = np.random.choice(self.mem_used, min(self.mem_used, self.eps_mem_batch), replace=False)
        # er_mem_indices = torch.from_numpy(er_mem_indices).to(x.device).long()
        unique_task = torch.unique(task)
        loss = 0
        for i in range(unique_task.shape[0]):
            ttask = unique_task[i]
            tx = x[task == ttask]
            ty = y[task == ttask]
            tt = task[task == ttask]
            y_hat = self(tx, tt)
            loss += self.criterion(y_hat, ty)

        self.log('train_loss', loss)
        return loss

    def train(self, train_loader, val_loader, task_num=0, ref_model=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.train()
        opt = self.configure_optimizers()
        # print('<================== Task {} ===================>'.format(task_num))
        best_Acc = 0
        for e in range(20):
            tot_loss = 0
            tot_size = 0
            end = time.time()
            for x, y, task_id in train_loader:
                x = x.to(device)
                y = y.to(device)
                task_id = task_id.to(device)
                y_hat = self(x, task_id)
                loss = self.criterion(y_hat, y)
                if ref_model is not None:
                    mm_loss = self.similarity(ref_model)
                    loss += mm_loss
                    # fusing feature manifold in replay
                    er_mem_indices = np.random.choice(self.mem_used, min(self.mem_used, self.eps_mem_batch),
                                                      replace=False)
                    er_mem_indices = torch.from_numpy(er_mem_indices).to(x.device).long()
                    # er_mem_indices = torch.from_numpy(er_mem_indices).long()
                    old_x = self.mem_data[er_mem_indices]
                    old_y = self.mem_label[er_mem_indices]
                    old_task = self.mem_task_id[er_mem_indices]
                    #
                    ref_index = ((old_task - 1) * old_x.shape[0] + torch.Tensor(range(old_x.shape[0])).to(
                        old_x.device)).long()

                    old_feature = self.model.features(old_x)
                    old_y_pred = self.model.logits(old_feature)[ref_index]
                    # old_y_pred = self(old_x, old_task)
                    loss += self.criterion(old_y_pred, old_y)
                    # ref_index = ((old_task - 1) * old_x.shape[0] + torch.Tensor(range(old_x.shape[0])).to(old_x.device)).long()
                    with torch.no_grad():
                        # old_y_ref_hat = ref_model(old_x)[ref_index]
                        old_ref_feature = ref_model.features(old_x)
                    # loss += self.feature_matching_loss(old_y_ref_hat,old_y_pred)
                    loss += self.feature_matching_loss(old_feature, old_ref_feature)
                opt.zero_grad()
                loss.backward()

                tot_loss += loss.item()
                tot_size += x.shape[0]
                opt.step()
            Acc = self.validation(val_loader)
            time_cost = time.time() - end
            if ref_model is not None:
                print(
                    f'<=== epoch: {e}   loss: {tot_loss / tot_size}   Accuracy: {Acc}   time: {time_cost}  similarity: {self.similarity(ref_model)}===>')
                # print(f'<=== epoch: {e}    loss: {tot_loss / tot_size}   time: {time_cost}  matching_loss: {self.manifold_matching_loss(ref_model)} ===>')
            else:
                print(f'<=== epoch: {e}   loss: {tot_loss / tot_size}   Accuracy: {Acc}   time: {time_cost} ===>')
            if Acc > best_Acc:
                torch.save(self.model.state_dict(),
                           r'checkpoint/resnet_best_parameters' + str(self.episodic_mem_size) + '.pkl')
                best_Acc = Acc
        # Model fusion
        # print(f'Acc: {self.validation(val_loader)}')
        # if ref_model is not None:
        #     self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[ref_model,self.model])
        # self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[ref_model, self.model])
        #
        # if ref_model is not None:
        #     param_dict = dict(ref_model.named_parameters())
        #     for name, p in self.model.named_parameters():
        #         p.data = 0.2 * param_dict[name].data + 0.8 * p.data

        self.update_memory(train_loader)
        torch.save(self.model.state_dict(), r'checkpoint/resnet_best_parameters' + str(self.episodic_mem_size) + '.pkl')
        return best_Acc

    def validation_step(self, batch, batch_idx):
        x, y, task = batch
        y_hat = self(x, task)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def accuracy(self, output, target):
        return (output.argmax(dim=1) == target).float().mean().item()

    def validation(self, val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        cnt, acc = 0, 0
        for x, y, name in val_loader:
            x = x.to(device)
            y = y.to(device)
            name = name.to(device)
            with torch.no_grad():
                y_hat = self(x, name)
                acc += (y_hat.argmax(dim=1) == y).float().sum().item()
                # acc += self.accuracy(y_hat,y)
                cnt += y_hat.shape[0]
        return acc / cnt

    def update_memory(self, dataset_loader):
        self.model.eval()
        replay_feature, center, distance, sorted_index = None, None, None, None
        for x_batch, y_batch, task_batch in dataset_loader:
            for index in range(x_batch.shape[0]):
                x, y, t = x_batch[index].cuda(), y_batch[index].cuda(), task_batch[index].cuda()
                if self.episodic_mem_size > self.mem_used:
                    self.mem_data[self.mem_used] = x
                    self.mem_label[self.mem_used] = y
                    self.mem_task_id[self.mem_used] = t
                    self.mem_used += 1
                else:
                    if center == None:
                        ######## AlexNet
                        replay_feature = self.model.features(self.mem_data)
                        replay_feature = F.normalize(replay_feature, dim=1)
                        ########
                        ######## resnet
                        # replay_feature = F.adaptive_avg_pool2d(F.relu(self.model.features(self.mem_data)), 1)
                        # replay_feature = replay_feature.view(replay_feature.size(0), -1)
                        # replay_feature = F.normalize(replay_feature, dim=1)
                        ########
                        center = F.normalize(replay_feature.mean(dim=0, keepdim=True), dim=1)
                        # replay_feature = torch.cat([self(self.mem_data[i * 10: (i + 1) * 10], self.mem_task_id[i * 10: (i + 1) * 10]) for i in range(0,self.episodic_mem_size // 10)], dim = 0)
                        # center = replay_feature.mean(dim=0, keepdim=True)
                        # distance = self.Sphere_dist(center, replay_feature).squeeze()
                        distance = self.pairwise_distances(center, replay_feature).squeeze()
                        sorted_index = torch.argsort(distance)
                    x = x.unsqueeze(0)
                    t = t.unsqueeze(0)

                    ####### AlexNet
                    cur_feature = self.model.features(x)
                    cur_feature = F.normalize(cur_feature, dim=1)
                    ######## resnet
                    # cur_feature = F.adaptive_avg_pool2d(F.relu(self.model.features(x)), 1)
                    # cur_feature = cur_feature.view(cur_feature.size(0), -1)
                    ########
                    # cur_feature = self.model.logits(cur_feature)[t]
                    # cur_feature = self(x,t)
                    # dis_to_center = torch.dist(F.normalize(cur_feature, dim = 1), center, 2) ** 2
                    dis_to_center = torch.dist(cur_feature, center, 2) ** 2
                    if dis_to_center > distance[sorted_index[-1]]:
                        j = np.random.randint(0, self.mem_used)
                        self.mem_data[j], self.mem_label[j], self.mem_task_id[j] = x, y, t
                        distance[j] = dis_to_center
                        sorted_index = torch.argsort(distance)
                    # elif dis_to_center >= distance[sorted_index[0]]:
                    else:
                        j = np.random.randint(0, self.data_seen)
                        # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
                        if j < self.episodic_mem_size:
                            self.mem_data[sorted_index[j]], self.mem_label[sorted_index[j]], self.mem_task_id[
                                sorted_index[j]] = x, y, t
                            distance[sorted_index[j]] = dis_to_center
                            sorted_index = torch.argsort(distance)

                self.data_seen += 1

    def reservior_update(self, dataset):

        for x, y, task in dataset:
            self.update_reservior(x, y, task)

    def pairwise_distances(self, x, y):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
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

    def feature_matching_loss(self, X, Y, eps=1e-9):
        X, Y = X.view(X.shape[0], -1), Y.view(Y.shape[0], -1)
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        # geodesic = self.Sphere_dist(X,Y).trace() / X.shape[0]
        # geodesic = torch.diag(self.Sphere_dist(X,Y)).mean()
        geodesic = torch.diag(self.pairwise_distances(X, Y)).mean()
        return geodesic

    def update_reservior(self, current_image, current_label, current_task):
        """
        Update the episodic memory with current example using the reservior sampling
        """
        if self.episodic_mem_size > self.data_seen:
            self.mem_data[self.data_seen] = current_image
            self.mem_label[self.data_seen] = current_label
            self.mem_task_id[self.data_seen] = current_task
            self.mem_used += 1
        else:
            j = np.random.randint(0, self.data_seen)
            if j < self.episodic_mem_size:
                self.mem_data[j] = current_image
                self.mem_label[j] = current_label
                self.mem_task_id[j] = current_task
        self.data_seen += 1



