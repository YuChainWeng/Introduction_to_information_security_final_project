import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import scipy.sparse as sp
import numpy as np
from torch_sparse import SparseTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn as nn

class EstimateAdj(nn.Module):
    """
    只參數化「邊權值向量」── indices 是 buffer，不佔參數；forward() 組回稀疏 COO。
    """
    def __init__(self, adj_sp: torch.Tensor, symmetric: bool = False, device: str = "cuda"):
        """
        Parameters
        ----------
        adj_sp : torch.sparse_coo_tensor  (必須 coalesce)
        """
        super().__init__()
        adj_sp = adj_sp.coalesce().to(device)

        # 把 COO indices 固定成 buffer
        self.register_buffer("indices", adj_sp.indices())      # shape = (2, m)
        # m 個可學參數：初值全 1
        self.values = nn.Parameter(adj_sp.values().clone())    # shape = (m,)

        self.N = adj_sp.size(0)
        self.symmetric = symmetric
        self.device = device

    def forward(self) -> torch.Tensor:
        A = torch.sparse_coo_tensor(
            self.indices, self.values,
            (self.N, self.N),
            dtype=self.values.dtype, device=self.device
        )
        if self.symmetric:
            A = 0.5 * (A + A.t())
        return A.coalesce()        # 回傳稀疏 COO



class PGD(Optimizer):
    def __init__(self, params, proxs, alphas, lr, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    """Proximal Operators.
    """
    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm.
        """
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm).
        """
        device = data.device
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):
        device = data.device
        U, S, V = torch.svd(data)
        self.nuclear_norm = S.sum()
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]),range(0, U.shape[0])]).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V
prox_operators = ProxOperators()


def prox_nuclear(A_sp: SparseTensor, tau: float):
    """Proximal operator of the nuclear norm for a sparse tensor."""
    A = A_sp.to_dense()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    S = torch.clamp(S - tau, min=0.0)
    A_new = (U * S.unsqueeze(-2)) @ Vh
    return SparseTensor.from_dense(A_new).coalesce()
