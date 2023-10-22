import numpy as np
import torch.nn as nn
import torch

import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
# import pymanopt.solvers
import numpy as np
import torch
import os, tqdm
import csv
from tqdm import tqdm
import numpy.linalg as la

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def orthonormalize(vectors, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in tqdm(range(start_idx,vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)
    return vectors

def single_projection(previous_p, gamma,lamda,feature_dim = 256, rank = 2):
    P = [v for _, v in previous_p.items()]
    P = torch.cat(P,dim = 1)
    manifold = pymanopt.manifolds.Oblique(feature_dim, rank)
    def cost(point, P):
        inter_task = (P.T @ point) ** 2
        intra_task = (point.T @ point) ** 2
        return lamda * (inter_task.sum()) + gamma * (intra_task - np.identity(point.shape[1])).sum() / 2
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    solver = pymanopt.solvers.SteepestDescent()
    solution = solver.solve(problem)
    return solution

def oblique_projection_matrix(tasks_name, feature_dim = 256, min_dim = 0, num_tasks = None, multiple = 1):
    if num_tasks == None:
        num_tasks = len(tasks_name)
    anp.random.seed(42)
    # num_tasks = 95
    # rank = max(min_dim,feature_dim // num_tasks)
    # rank = 64 // num_tasks
    # rank = 256 // num_tasks
    rank = feature_dim // num_tasks
    manifold = pymanopt.manifolds.Oblique(feature_dim, num_tasks * rank)
    def cost(point):
        return ((((point.T @ point) - np.identity(point.shape[1])) ** 2)).sum()
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    solver = pymanopt.solvers.SteepestDescent(maxiter=200, minstepsize=1e-20)
    solution = solver.solve(problem)
    np.random.shuffle(solution)
    projections = nn.ParameterDict()
    for tt, task in enumerate(tasks_name):
        offset = tt * rank
        A = solution[:,offset:offset + rank]
        # A = np.concatenate((solution[:, offset:offset + rank], solution[:, feature_dim - share_dims:]), axis=1)
        proj = np.matmul(A, np.transpose(A))
        # 乘法实验
        # projections[str(task)] = nn.Parameter(torch.Tensor(proj) * multiple,requires_grad = False)
        # 除法实验
        projections[str(task)] = nn.Parameter(torch.Tensor(proj) / multiple,requires_grad = False)
        # if task > 1:
        #     projections[str(task)] = nn.Parameter(torch.Tensor(proj) + projections[str(task - 1)],requires_grad = False)
        # else:
        #     projections[str(task)] = nn.Parameter(torch.Tensor(proj), requires_grad=False)
    return projections

def generate_projection_matrix(tasks_name, feature_dim=256, share_dims=0, qr = True, num_tasks = None):
    """
    Project features (v) to subspace parametrized by A:

    Returns: A.(A^T.A)^-1.A^T
    """
    if num_tasks == None:
        num_tasks = len(tasks_name)
    # num_tasks = len(tasks_name)
    # num_tasks = 100
    rank = (feature_dim - share_dims) // num_tasks
    assert num_tasks * rank <= (feature_dim - share_dims), "Feature dimension should be less than num_tasks * rank"

    # Generate ONBs
    if qr:
        print('Generating ONBs from QR decomposition')
        rand_matrix = np.random.uniform(size=(feature_dim, feature_dim))
        q, r = np.linalg.qr(rand_matrix, mode='complete')
    else:
        print('Generating ONBs from Identity matrix')
        q = np.identity(feature_dim)
    projections = nn.ParameterDict()

    for tt, task in enumerate(tasks_name):
        offset = tt * rank
        A = np.concatenate((q[:, offset:offset + rank], q[:, feature_dim - share_dims:]), axis=1)
        proj = np.matmul(A, np.transpose(A))
        projections[str(task)] = nn.Parameter(torch.FloatTensor(proj),requires_grad = False)
    return projections

def unit_test_projection_matrices(projection_matrices):
    """
    Unit test for projection matrices
    """
    num_matrices = len(projection_matrices)
    feature_dim = projection_matrices['1'].shape[0]
    rand_vetcor = np.random.rand(1, feature_dim)
    projections = []
    for tt in projection_matrices.keys():
        print('Task:{}, Projection Dims: {}, Projection Rank: {}'.format(tt, projection_matrices[tt].shape, np.linalg.matrix_rank(projection_matrices[tt].numpy())))
        projections.append((np.squeeze(np.matmul(rand_vetcor, projection_matrices[tt].numpy()))))
    print('\n\n ******\n Sanity testing projections \n********')
    for i in range(num_matrices):
        for j in range(num_matrices):
            print('P{}.P{}={}'.format(i, j, np.dot(projections[i], projections[j])))


def deconv_orth_dist(kernel, stride=2, padding=1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1] / 2))
    target[:, :, ct, ct] = torch.eye(o_c).cuda()
    return torch.norm(output - target)


def orth_dist(mat, stride=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).cuda())