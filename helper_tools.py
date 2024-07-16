from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    """Get the preferred device:
    check if there is a GPU available:NVIDIA GPU  or MPS (Apple Silcon GPU) if available).
    """

    # check if there is nvidia or cuda gpu
    if torch.cuda.is_available():
        return torch.device("cuda")

    # check if there is an apple silicon gpu
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # otherwise use the cpu
    return torch.device("cpu")


def plot_loss_curve(d_losses: List[float], g_losses: List[float], EPOCHS: int) -> None:
    """Plot loss curve of critic and generator.

    Args:
        d_losses (List[float]): List of Discriminator losses.
        g_losses (List[float]): List of generator losses.
        EPOCHS (int): Total number of epochs.
    """
    # normalize losses

    # Assuming 'critic_losses' and 'generator_losses' are lists of recorded losses

    gen_losses = np.asarray(g_losses)
    critic_losses = np.asarray(d_losses)

    gen_losses = (gen_losses - gen_losses.min()) / (gen_losses.max() - gen_losses.min())
    critic_losses = np.asarray(d_losses)

    critic_losses = (critic_losses - critic_losses.min()) / (
        critic_losses.max() - critic_losses.min()
    )
    # plt.figure(figsize=(20, 10))
    plt.figure(figsize=(10, 5))
    x = range(1, EPOCHS + 1)
    plt.title("Normalized Loss Curves")
    plt.plot(x, critic_losses, color="red", label="Critic Loss")
    plt.plot(x, gen_losses, color="blue", label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Loss Curves")
    plt.legend()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()


def set_random_seeds(seed_value=42): 
    """
    Set random seeds for torch, numpy, and torch's cuDNN backend for reproducibility.
    
    Args:
        seed_value (int): The seed value to set for random number generation. Default is 42.
    """
    seed = seed_value
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True