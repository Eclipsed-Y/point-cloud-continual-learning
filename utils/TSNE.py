from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

import  torch
# ckpt_dir="../images"
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
def Draw(X, Y, model_name):
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(X)  # 将高维数据降维到二维平面可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y)
    # plt.legend()
    plt.savefig('images/{}.pdf'.format(model_name), dpi=600)

def Draw_dataset(dataloader, model_name):
    X, Y = None, None
    for x, y, task in dataloader:
        if X is None:
            X = x
        else:
            X = torch.cat((X, x), dim=0)
        if Y is None:
            Y = y
        else:
            Y = torch.cat((Y, y), dim=0)
    tot = args.total_classes * args.mem_size_per_class
    TSNE.Draw(agent.model.features(X.cuda()).cpu().detach().numpy(), Y.detach().numpy(), f'ER_{tot}')