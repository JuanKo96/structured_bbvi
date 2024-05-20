import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_negative_energy(target_dist, samples):
    return -torch.mean(
        target_dist.log_prob(samples)
    )


def proximal_update(param, gamma):
    delta_L = 0.5 * (torch.sqrt(param.data**2 + 4 * gamma) - param.data)
    param.data += delta_L

def get_target_posterior(config, device, seed, jitter):
    set_seed(seed)
    d_z = config.d_z
    d_y = config.d_y
    N = config.N
    jitter = float(jitter)
    m = torch.randn(d_z + N * d_y, device=device).double()
    Lz = init_tril_with_positive_diag(d_z, d_z, device=device)
    Ly = [
            init_tril_with_positive_diag(d_y, d_y, device=device)
            for _ in range(N)
        ]
    Lyz = [torch.randn(d_y, d_z, device=device).double() for _ in range(N)]
    
    d_total = d_z + N * d_y
    L_dense = torch.zeros((d_total, d_total), device=m.device).double()
    Lz = Lz.tril()
    Ly = [ly.tril() for ly in Ly]
    L_dense[: d_z, : d_z] = Lz
    for n in range(N):
        start = d_z + n * d_y
        L_dense[start : start + d_y, : d_z] = Lyz[n]
        L_dense[start : start + d_y, start : start + d_y] = Ly[n]
    # L_dense = L_dense + torch.eye(d_total, device=m.device) * jitter
    L_dense = L_dense + torch.diag(torch.FloatTensor(d_total).uniform_(1, 2)).double().to(m.device)
    cov = (
        L_dense @ L_dense.T #+ torch.eye(d_total, device=m.device) * jitter
    )
    # L = torch.linalg.cholesky(cov)
    return m, cov, L_dense

def init_tril_with_positive_diag(rows, cols, device):
    tril = torch.tril(torch.randn(rows, cols, device=device).double())
    tril.diagonal().uniform_(0.1, 1.0).double()
    return tril