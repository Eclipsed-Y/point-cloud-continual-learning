import os

SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import argparse
import torch
from random import shuffle
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import methods
import numpy as np
import os
from tqdm import tqdm
import csv, copy
from torch.utils.data import DataLoader

result_root_path = r'./result'


# 构建超参数
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
    parser.add_argument('--run_seed', type=int, default=2023)
    parser.add_argument('--num_task', type=int, default=8)
    parser.add_argument('--categories', type=int,
                        default=['table', 'chair', 'airplane', 'car', 'sofa', 'lamp', 'vessel', 'cabinet', 'monitor',
                                 'guitar', 'bookshelf', 'laptop', 'pistol', 'bed', 'rocket', 'bag', 'earphone', 'cap'])
    parser.add_argument('--dataset', type=str, default='ShapeNet')
    parser.add_argument('--dataroot', type=str, default='./pointCloud/data/shapenet.hdf5')
    parser.add_argument('--scale_mode', type=str, default='shape_unit',
                        help='global_unit, shape_unit, shape_bbox, shape_half, shape_34')
    parser.add_argument('--transform', type=str, default=None)
    parser.add_argument('--mem_size', type=int, default=256)
    parser.add_argument('--mem_batch', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--eps_mem_batch', type=int, default=16)

    args = parser.parse_args([])
    return args


args = init_args()


# 创建点云训练集和验证集
def ShapeNet(dataroot, train_categories, val_categories, scale_mode, transform):
    from pointCloud.utils.dataset import ShapeNetCore
    # 加载训练数据集
    train_dset = ShapeNetCore(
        path=dataroot,
        cates=train_categories,
        split='train',
        scale_mode=scale_mode,
        transform=transform,
    )
    # 加载验证数据集`
    val_dset = ShapeNetCore(
        path=dataroot,
        cates=val_categories,
        split='val',
        scale_mode=scale_mode,
        transform=transform,
    )

    return train_dset, val_dset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


setup_seed(1016)

if __name__ == '__main__':
    agent = None
    for n_task in range(0, args.num_task):
        if n_task == 0:
            train_now_categories = args.categories[0: 4]  # 起始为四分类任务

        else:
            train_now_categories = args.categories[2 + n_task * 2: 4 + n_task * 2]  # 后续每次为一个二分类任务

        val_now_categories = args.categories[0: n_task * 2 + 4]

        outdim = 2 * (n_task + 2)  # 模型输出层神经元数量
        train_dataset, val_dataset = ShapeNet(args.dataroot, train_now_categories, val_now_categories, args.scale_mode,
                                              args.transform)  # 创建数据集
        if n_task == 0:
            agent = methods.MLP_correlation.MLP_correlation(args, outdim)  # 创建模型
            print(f"模型创建成功，当前输出层为{agent.model.out_dim}")
        else:
            agent.model.add_output_layer(2)  # 改变输出层神经元数量
            print(f"模型输出层改变成功，当前输出层为{agent.model.out_dim}")
        print(f'任务{n_task},类别为：{train_now_categories}')  # 输出当前任务类别

        dir_path = f'checkpoint/pointcloud/task_{n_task}/'
        if not os.path.exists(dir_path):  # 如果没有当前文件夹，则创建一个
            os.makedirs(dir_path)
        print(f'====================== Task {n_task} =======================')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=0)
        if n_task == 0:
            Acc = agent.train(train_loader, val_loader, task_num=n_task)
        else:
            ref_model = copy.deepcopy(agent.model)  # 将当前model拷贝到ref_model上
            for p in ref_model.parameters():
                p.requires_grad = False  # 冻结ref_model的参数更新
            Acc = agent.train(train_loader, val_loader, task_num=n_task, ref_model=ref_model)
        print('task : {}  Acc : {}'.format(n_task, Acc))
