import torch
import torch.nn as nn
from utils import proximal_update


class DiagonalVariational(nn.Module):
    def __init__(self, d, n_sample, jitter):
        super(DiagonalVariational, self).__init__()
        self.m = nn.Parameter(torch.zeros(d).double())
        self.diag_L = nn.Parameter(torch.ones(d).double())
        self.n_sample = n_sample
        self.jitter = float(jitter)

    def forward(self):
        diag_L = self.diag_L
        L = torch.diag(diag_L)
        L = L + torch.eye(L.size(0), device=L.device) * self.jitter
        std_normal = torch.randn(self.n_sample, len(self.m), device=L.device).double()
        z = self.m + std_normal @ L
        return z


class FullRankVariational(nn.Module):
    def __init__(self, d, n_sample, jitter):
        super(FullRankVariational, self).__init__()
        self.m = nn.Parameter(torch.zeros(d).double())
        self.L = nn.Parameter(torch.eye(d).double())
        self.n_sample = n_sample
        self.jitter = float(jitter)

    def init_tril_with_positive_diag(self, rows, cols):
        tril = torch.tril(torch.randn(rows, cols).double())
        tril.diagonal().uniform_(0.1, 1.0).double()
        return tril

    def forward(self):
        L = torch.tril(self.L)
        L = L + torch.eye(L.size(0), device=L.device) * self.jitter
        std_normal = torch.randn(self.n_sample, len(self.m), device=L.device).double()
        z = self.m + std_normal @ L
        return z


class StructuredVariational(nn.Module):
    def __init__(self, d_z, d_y, N, n_sample, jitter):
        super(StructuredVariational, self).__init__()
        self.d_z = d_z
        self.d_y = d_y
        self.N = N
        self.n_sample = n_sample
        self.jitter = float(jitter)
        self.m = nn.Parameter(torch.zeros(d_z + N * d_y).double())
        Lz = torch.eye(d_z).double()
        Ly_blocks = torch.eye(d_y).repeat(N, 1, 1).double()
        Lyz = torch.zeros(N * d_y, d_z).double()

        d_total = self.d_z + self.N * self.d_y
        indices = []

        # Lz indices
        for i in range(self.d_z):
            for j in range(i + 1):
                indices.append([i, j])

        # Ly indices
        for n in range(self.N):
            for i in range(self.d_y):
                for j in range(i + 1):
                    indices.append(
                        [self.d_z + n * self.d_y + i, self.d_z + n * self.d_y + j]
                    )

        # Lyz indices
        for n in range(self.N):
            for i in range(self.d_y):
                for j in range(self.d_z):
                    indices.append([self.d_z + n * self.d_y + i, j])

        self.indices = torch.tensor(indices).t()
        self.d_total = d_total

        # Precompute flattened values parts
        self.Lz_flat = nn.Parameter(
            torch.cat([Lz[i, : i + 1] for i in range(self.d_z)])
        ).double()
        self.Lyz_flat = nn.Parameter(Lyz.flatten()).double()
        self.Ly_blocks_flat = nn.Parameter(
            torch.cat(
                [
                    Ly_blocks[n].flatten()[
                        torch.tril(torch.ones(self.d_y, self.d_y)).flatten().bool()
                    ]
                    for n in range(self.N)
                ]
            )
        ).double()
        # Precompute diagonal indices
        self.Lz_diag_indices = torch.cat(
            [Lz[i, : i + 1] for i in range(self.d_z)]
        ).bool()
        self.Ly_blocks_diag_indices = torch.cat(
            [
                Ly_blocks[n].flatten()[
                    torch.tril(torch.ones(self.d_y, self.d_y)).flatten().bool()
                ]
                for n in range(self.N)
            ]
        ).bool()

    def construct_matrix(self):
        d_total = self.d_total
        device = self.m.device
        indices = self.indices.to(device)

        # Construct values by concatenating precomputed parts and current parameters
        values = torch.cat([self.Lz_flat, self.Ly_blocks_flat, self.Lyz_flat]).to(
            device
        )

        L_sparse = torch.sparse_coo_tensor(
            indices, values, torch.Size([d_total, d_total]), device=device
        )
        L_dense = L_sparse.to_dense()

        return L_dense

    def forward(self):
        d_total = self.d_total
        device = self.m.device
        L_dense = self.construct_matrix()

        std_normal = torch.randn(self.n_sample, d_total, device=device).double()
        z = self.m + std_normal @ L_dense.T
        return z

    def proximal_update_step(self, gamma):
        # Update the diagonal of Lz
        Lz_diag_indices = self.Lz_diag_indices.to(self.Lz_flat.device)
        self.Lz_flat.data[Lz_diag_indices] += 0.5 * (
            torch.sqrt(self.Lz_flat.data[Lz_diag_indices] ** 2 + 4 * gamma)
            - self.Lz_flat.data[Lz_diag_indices]
        )

        # Update the diagonal of Ly blocks
        Ly_blocks_diag_indices = self.Ly_blocks_diag_indices.to(
            self.Ly_blocks_flat.device
        )
        self.Ly_blocks_flat.data[Ly_blocks_diag_indices] += 0.5 * (
            torch.sqrt(
                self.Ly_blocks_flat.data[Ly_blocks_diag_indices] ** 2 + 4 * gamma
            )
            - self.Ly_blocks_flat.data[Ly_blocks_diag_indices]
        )
