# -*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.11.24
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import PCA

'''
主成分分析_2维数据降维1维演示函数
'''




'''主成分分析_PCA图像数据降维'''


def PCA_faceImage(X):
    m = X.shape[0]  # 数据条数

    X_norm, mu, sigma = featureNormalize(X)  # 归一化

    Sigma = torch.mm(X_norm.T, X_norm) / m  # 求Sigma(协方差矩阵)
    # Sigma = np.cov(X_norm.T) 也可以直接求
    U, S, V = torch.svd(Sigma)
    # U, S, V = np.linalg.svd(Sigma)  # 奇异值分解

    K = 128  # 降维128维(原先是32*32=1024维的)
    Z = projectData(X_norm, U, K)

    # X_rec = recoverData(Z, U, K)  # 恢复数据
    # display_imageData(X_rec[0:100, :])
    return Z

# 可视化二维数据
def plot_data_2d(X, marker):
    plt.plot(X[:, 0], X[:, 1], marker)
    return plt


# 归一化数据
def featureNormalize(X):
    '''（每一个数据-当前列的均值）/当前列的标准差'''
    n = X.shape[1]
    mu = torch.mean(X, dim=0)  # axis=0表示列
    sigma = torch.std(X, dim=0)
    X = (X - mu) / sigma
    # for i in range(n):
    #     X[:, i] = (X[:, i] - mu[i]) / sigma[i]
    return X, mu, sigma


# 映射数据
def projectData(X_norm, U, K):
    # Z = np.zeros((X_norm.shape[0], K))

    U_reduce = U[:, :K]  # 取前K个
    Z = torch.mm(X_norm, U_reduce)
    return Z

# 恢复数据
def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_recude = U[:, 0:K]
    X_rec = np.dot(Z, np.transpose(U_recude))  # 还原数据（近似）
    return X_rec


# 显示图片
def display_imageData(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的图片整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    m, n = imgData.shape
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n / width);
    rows_count = np.int32(np.floor(np.sqrt(m)))
    cols_count = np.int32(np.ceil(m / rows_count))
    pad = 1
    display_array = -np.ones((pad + rows_count * (height + pad), pad + cols_count * (width + pad)))
    for i in range(rows_count):
        for j in range(cols_count):
            max_val = np.max(np.abs(imgData[sum, :]))
            display_array[pad + i * (height + pad):pad + i * (height + pad) + height,
            pad + j * (width + pad):pad + j * (width + pad) + width] = imgData[sum, :].reshape(height, width,
                                                                                               order="F") / max_val  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.show()

