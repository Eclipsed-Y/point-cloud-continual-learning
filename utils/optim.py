import torch
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from typing import List, Optional
from torch import Tensor

class StiefelOpt(Optimizer):
    '''
        Caylay transformer
    '''
    def __init__(self,params,lr = required,momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(StiefelOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    # Pytorch的
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                g = d_p
                w = p.data
                shape = g.shape
                g = g.reshape(-1,shape[0])
                w = w.reshape(-1,shape[0])
                gxT = g @ w.T
                xTgxT = w.T @gxT
                A_hat = gxT - 0.5 * w @ xTgxT
                A = A_hat - A_hat.T
                U = A @ w

                t = 0.5 * 2 / (A.norm(1,1).max() + 1e-8)
                alpha = min(t,group['lr'])
                Y_0 = w - alpha * U
                Y_1 = w - alpha * (A @ (0.5 * (w + Y_0)))
                Y_2 = w - alpha * (A @ (0.5 * (w + Y_1)))
                p.data = Y_2.reshape(shape)
                # p.data.add_(-group['lr'], d_p)

        return loss

class ObliqueOpt(Optimizer):
    '''
        Caylay transformer
    '''
    def __init__(self,params,lr = required,momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ObliqueOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _normalize_columns(self, X):
        """Return an l2-column-normalized ablation_study_copy of the matrix X."""
        X = X.T
        return (X / (X.norm(dim = 0, keepdim = True))).T
        # return X
        # return X / la.norm(X, axis=0)[np.newaxis, :]

    def retr(self, X, U):
        return self._normalize_columns(X + U)

    def proj(self, X, H):
        X = X.T
        H = H.T
        return (H - X * ((X * H).sum(dim = 0, keepdim = True))).T

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    # Pytorch的
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                d_p = self.proj(p.data, d_p)
                p.data = self.retr(p.data,d_p * -lr)
                # p.data.add_(-group['lr'], d_p)
        return loss