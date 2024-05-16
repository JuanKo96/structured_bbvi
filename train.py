from tqdm import trange
import torch
import wandb
from utils import compute_negative_energy, proximal_update


def train(q, optimizer, config, device, seed, step_size):
    progress_bar = trange(config.n_iterations, desc="Optimizing", leave=True)
    for iteration in progress_bar:
        optimizer.zero_grad()
        samples = q()
        loss = compute_negative_energy(samples)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if config.model_type == "DiagonalVariational":
                proximal_update(q.log_diag_L, config.gamma)
            elif config.model_type == "FullRankVariational":
                for i in range(q.L.size(0)):
                    proximal_update(q.L[i, i], config.gamma)
            elif config.model_type == "StructuredVariational":
                for i in range(q.Lz.size(0)):
                    proximal_update(q.Lz[i, i], config.gamma)
                for n in range(config.N):
                    for i in range(q.Ly[n].size(0)):
                        proximal_update(q.Ly[n][i, i], config.gamma)

        progress_bar.set_postfix(loss=loss.item())

        # Log the loss to wandb
        wandb.log(
            {
                "iteration": iteration,
                "loss": loss.item(),
                "seed": seed,
                "step_size": step_size,
                "model_type": config.model_type,
            }
        )

    return loss.item()
