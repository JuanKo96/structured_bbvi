from tqdm import trange
import torch
import wandb
from utils import compute_negative_energy, proximal_update


def train(
    q,
    optimizer,
    config,
    seed,
    step_size,
    mu,
    L_Sigma,
    sigma_scalar,
):
    progress_bar = trange(config.n_iterations, desc="Optimizing", leave=True)
    hit_epsilon = False
    for iteration in progress_bar:
        optimizer.zero_grad()
        samples = q()
        loss = compute_negative_energy(samples, mu, sigma_scalar)
        # Sometimes, the loss gets negative values
        # loss = torch.clamp(loss, min=0, max=1e6)
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
                q.proximal_update_step(config.gamma)
                m = q.m
                L_var = q.construct_matrix()

        optimality_gap_m = torch.norm(m - mu) ** 2
        optimality_gap_C = torch.norm(L_var - L_Sigma, p="fro") ** 2
        optimality_gap = optimality_gap_m + optimality_gap_C

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
        if optimality_gap.item() < config.epsilon:
            if not hit_epsilon:
                wandb.log({"T_epsilon": iteration})
                hit_epsilon = True
        if iteration == config.n_iterations -1:
            if not hit_epsilon:
                wandb.log({"T_epsilon": iteration})
                hit_epsilon = True
    return loss.item(), optimality_gap_C.item(), optimality_gap.item()
