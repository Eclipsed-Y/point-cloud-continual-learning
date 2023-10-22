import torch

def margin_loss(pre_head, now_head):
    loss = 0.0
    num_pre_head = len(pre_head)
    num_now_head = len(now_head)
    # 计算参数之间的内积，目标是使它们尽量正交
    for i in range(num_pre_head):
        for j in range(num_now_head):
            param_i = pre_head[i].weight.view(-1)
            param_j = now_head[j].weight.view(-1)
            loss += torch.abs(torch.dot(param_i, param_j))

    return loss
