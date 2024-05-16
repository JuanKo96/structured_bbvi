import torch
import torch.nn as nn


class DiagonalVariational(nn.Module):
    def __init__(self, d, n_sample, jitter):
        super(DiagonalVariational, self).__init__()
        self.m = nn.Parameter(torch.randn(d).double())
        self.log_diag_L = nn.Parameter(torch.randn(d).double())
        self.n_sample = n_sample
        self.jitter = float(jitter)

    def forward(self):
        diag_L = self.log_diag_L
        L = torch.diag(diag_L)
        cov = L @ L.T + torch.eye(L.size(0), device=L.device) * self.jitter
        return torch.distributions.MultivariateNormal(self.m, cov).rsample(
            (self.n_sample,)
        )


class FullRankVariational(nn.Module):
    def __init__(self, d, n_sample, jitter):
        super(FullRankVariational, self).__init__()
        self.m = nn.Parameter(torch.randn(d).double())
        self.L = nn.Parameter(self.init_tril_with_positive_diag(d, d))
        self.n_sample = n_sample
        self.jitter = float(jitter)
        
    def init_tril_with_positive_diag(self, rows, cols):
        tril = torch.tril(torch.randn(rows, cols).double())
        tril.diagonal().uniform_(0.1, 1.0).double()
        return tril

    def forward(self):
        L = torch.tril(self.L)
        cov = L @ L.T + torch.eye(L.size(0), device=L.device) * self.jitter
        return torch.distributions.MultivariateNormal(self.m, cov).rsample(
            (self.n_sample,)
        )


class StructuredVariational(nn.Module):
    def __init__(self, d_z, d_y, N, n_sample, jitter):
        super(StructuredVariational, self).__init__()
        self.d_z = d_z
        self.d_y = d_y
        self.N = N
        self.n_sample = n_sample
        self.jitter = float(jitter)
        self.m = nn.Parameter(torch.randn(d_z + N * d_y).double())
        self.Lz = nn.Parameter(self.init_tril_with_positive_diag(d_z, d_z))
        self.Ly = nn.ParameterList(
            [
                nn.Parameter(self.init_tril_with_positive_diag(d_y, d_y))
                for _ in range(N)
            ]
        )
        self.Lyz = nn.ParameterList(
            [nn.Parameter(torch.randn(d_y, d_z).double()) for _ in range(N)]
        )

    def init_tril_with_positive_diag(self, rows, cols):
        tril = torch.tril(torch.randn(rows, cols).double())
        tril.diagonal().uniform_(0.1, 1.0).double()
        return tril

    def forward(self):
        d_total = self.d_z + self.N * self.d_y
        L_dense = torch.zeros((d_total, d_total), device=self.m.device).double()
        Lz = self.Lz.tril()
        Ly = [ly.tril() for ly in self.Ly]
        L_dense[: self.d_z, : self.d_z] = Lz
        for n in range(self.N):
            start = self.d_z + n * self.d_y
            L_dense[start : start + self.d_y, : self.d_z] = self.Lyz[n]
            L_dense[start : start + self.d_y, start : start + self.d_y] = Ly[n]
        cov = (
            L_dense @ L_dense.T + torch.eye(d_total, device=self.m.device) * self.jitter
        )
        return torch.distributions.MultivariateNormal(self.m, cov).rsample(
            (self.n_sample,)
        )
