from tqdm import trange
import torch
import wandb
from utils import compute_negative_energy, proximal_update


def train(q, optimizer, config, device, seed, step_size, target_dist, mu, L_Sigma):
    progress_bar = trange(config.n_iterations, desc="Optimizing", leave=True)
    for iteration in progress_bar:
        optimizer.zero_grad()
        samples = q()
        loss = compute_negative_energy(target_dist, samples)
        # Sometimes, the loss gets negative values
        loss = torch.clamp(loss, min=0, max=1e6)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if config.model_type == "DiagonalVariational":
                proximal_update(q.diag_L, config.gamma)
                
                m = q.m
                L_var = torch.diag(q.diag_L)
            elif config.model_type == "FullRankVariational":
                proximal_update(torch.diagonal(q.L), config.gamma)
                
                m = q.m
                L_var = torch.tril(q.L)
            elif config.model_type == "StructuredVariational":
                proximal_update(torch.diagonal(q.Lz), config.gamma)
                # proximal_update(q.Ly.diag(), config.gamma)
                # for block in q.Ly_blocks:
                #     proximal_update(block.diag(), config.gamma)
                proximal_update(torch.diagonal(q.Ly_blocks, dim1=1, dim2=2), config.gamma)

                m=q.m
                d_total = q.d_z + q.N * q.d_y

                # Construct L_dense efficiently
                L_dense = torch.zeros((d_total, d_total), device=q.m.device).double()
                Lz = q.Lz.tril()
                # Ly = q.Ly.tril()
                Ly_blocks = q.Ly_blocks.tril()
                Ly = torch.block_diag(*Ly_blocks)

                L_dense[:q.d_z, :q.d_z] = Lz
                L_dense[q.d_z:, :q.d_z] = q.Lyz
                L_dense[q.d_z:, q.d_z:] = Ly
                L_var = L_dense
                
        optimality_gap_m = torch.norm(m - mu)**2 
        optimality_gap_C = torch.norm(L_var - L_Sigma, p='fro')**2
        optimality_gap = optimality_gap_m +  optimality_gap_C

        progress_bar.set_postfix(loss=loss.item(), optimality_gap=optimality_gap.item())
        # progress_bar.set_postfix(loss=loss.item())

        # Log the loss to wandb
        wandb.log(
            {
                "iteration": iteration,
                "loss": loss.item(),
                "optimality_gap_m": optimality_gap_m.item(),
                "optimality_gap_C": optimality_gap_C.item(),
                "optimality_gap": optimality_gap.item(),
                "seed": seed,
                "step_size": step_size,
                "model_type": config.model_type,
            }
        )

    return loss.item(), optimality_gap_C.item(), optimality_gap.item()
