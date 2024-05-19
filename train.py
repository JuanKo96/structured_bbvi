from tqdm import trange
import torch
import wandb
from utils import compute_negative_energy, proximal_update


def train(q, optimizer, config, device, seed, step_size, mu, Sigma):
    progress_bar = trange(config.n_iterations, desc="Optimizing", leave=True)
    for iteration in progress_bar:
        optimizer.zero_grad()
        samples = q()
        loss = compute_negative_energy(samples, mu, Sigma)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if config.model_type == "DiagonalVariational":
                proximal_update(q.diag_L, config.gamma)
                m = q.m
                C = torch.diag(q.diag_L ** 2)
            elif config.model_type == "FullRankVariational":
                for i in range(q.L.size(0)):
                    proximal_update(q.L[i, i], config.gamma)
                m = q.m
                L = torch.tril(q.L)
                C = L @ L.T
            elif config.model_type == "StructuredVariational":
                for i in range(q.Lz.size(0)):
                    proximal_update(q.Lz[i, i], config.gamma)
                for n in range(config.N):
                    for i in range(q.Ly[n].size(0)):
                        proximal_update(q.Ly[n][i, i], config.gamma)
                m=q.m
                d_total = q.d_z + q.N * q.d_y
                L_dense = torch.zeros((d_total, d_total), device=q.m.device).double()
                Lz = q.Lz.tril()
                Ly = [ly.tril() for ly in q.Ly]
                L_dense[: q.d_z, : q.d_z] = Lz
                for n in range(q.N):
                    start = q.d_z + n * q.d_y
                    L_dense[start : start + q.d_y, : q.d_z] = q.Lyz[n]
                    L_dense[start : start + q.d_y, start : start + q.d_y] = Ly[n]
                C = (
                    L_dense @ L_dense.T + torch.eye(d_total, device=q.m.device) * q.jitter
                )
                
        optimality_gap = torch.norm(m - mu)**2 + torch.norm(C - Sigma, p='fro')**2

        progress_bar.set_postfix(loss=loss.item(), optimality_gap=optimality_gap.item())
        # progress_bar.set_postfix(loss=loss.item())

        # Log the loss to wandb
        wandb.log(
            {
                "iteration": iteration,
                "loss": loss.item(),
                "optimality_gap": optimality_gap.item(),
                "seed": seed,
                "step_size": step_size,
                "model_type": config.model_type,
            }
        )

    return loss.item()
