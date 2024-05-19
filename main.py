import torch
import torch.optim as optim
import argparse
import yaml
import wandb
import numpy as np
import os
from models import DiagonalVariational, FullRankVariational, StructuredVariational
from train import train
from utils import set_seed
from attrdict import AttrDict


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", default="config.yaml"
    )
    args, remaining_args = parser.parse_known_args()

    with open(args.config, "r") as f:
        default_args = yaml.load(f, Loader=yaml.FullLoader)

    argparser = argparse.ArgumentParser()
    for arg, val in default_args.items():
        if isinstance(val, list):
            argparser.add_argument(
                f"--{arg}", default=val, type=type(val[0]), nargs="+"
            )
        else:
            argparser.add_argument(f"--{arg}", default=val, type=type(val))

    parsed_args = argparser.parse_args(remaining_args)

    return AttrDict(vars(parsed_args))


def main(config):
    wandb_mode = config.get("wandb_mode", "online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if isinstance(config.step_sizes, list):
        step_sizes = config.step_sizes
    elif isinstance(config.step_sizes, dict):
        step_sizes = np.logspace(
            config.step_sizes["start"],
            config.step_sizes["stop"],
            config.step_sizes["num"],
        ).tolist()
    else:
        raise ValueError("Invalid format for step_sizes")

    d_total = config.d_z + config.N * config.d_y
    
    set_seed(1234)
    # Set random mu and Sigma
    mu = torch.randn(d_total, device=device).double()
    # mu = torch.zeros(d_total, device=device).double()
    # L_Sigma = torch.tril(torch.randn(d_total, d_total, device=device).double())
    # L_Sigma.diagonal().uniform_(0.1, 1.0)  # Ensure positive diagonal elements
    # L_Sigma = L_Sigma
    # Sigma = L_Sigma @ L_Sigma.T + torch.eye(L_Sigma.size(0), device=L_Sigma.device) * 1e-1
    # L_Sigma = torch.linalg.cholesky(Sigma)
    Sigma = torch.eye(d_total, device=device).double() * 1e-2
    L_Sigma = torch.linalg.cholesky(Sigma)
    
    # data = torch.randn(config.n_sample, d_total, device=device).double()
    # mu = torch.mean(data, dim=0).double()
    # cov = torch.cov(data.T)
    # # Add jitter for positive definiteness
    # Sigma = cov + torch.eye(cov.size(0), device=cov.device) * 1e-6
    
    for step_size in step_sizes:
        # for seed in config.seeds:
        seed = config.seed
        set_seed(seed)
        

        if config.model_type == "DiagonalVariational":
            q = DiagonalVariational(d_total, config.n_sample, config.jitter).to(device)
        elif config.model_type == "FullRankVariational":
            q = FullRankVariational(d_total, config.n_sample, config.jitter).to(device)
        elif config.model_type == "StructuredVariational":
            q = StructuredVariational(
                config.d_z, config.d_y, config.N, config.n_sample, config.jitter
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        optimizer = optim.SGD(q.parameters(), lr=step_size)

        run = wandb.init(
            project=f"{config.wandb_project}",
            config=config,
            mode=wandb_mode,
            reinit=True,
        )
        wandb.config.update(
            {"seed": seed, "step_size": step_size}, allow_val_change=True
        )

        final_loss = train(q, optimizer, config, device, seed, step_size, mu, Sigma, L_Sigma)

        wandb.log({"final_loss": final_loss})
        run.finish()


if __name__ == "__main__":
    config = parse_args_and_config()
    main(config)
