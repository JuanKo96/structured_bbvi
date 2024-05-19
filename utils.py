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


def compute_negative_energy(samples,mu, Sigma):
    return -torch.mean(
        torch.distributions.MultivariateNormal(mu, Sigma).log_prob(samples)
    )


def proximal_update(param, gamma):
    delta_L = 0.5 * (torch.sqrt(param.data**2 + 4 * gamma) - param.data)
    param.data += delta_L
