from collections import Counter
import random
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



def sample_equal_data(train_data, train_labels):
    sampled_train_data = []
    sampled_train_labels = []

    # Combine data and labels
    train_data_with_labels = list(zip(train_data, train_labels))
    # Shuffle data
    random.shuffle(train_data_with_labels)

    # Count classes
    class_counts = Counter(train_labels)
    min_class_count = min(class_counts.values())

    # Initialize a dictionary to keep track of sampled data for each class
    sampled_data_dict = {label: [] for label in class_counts.keys()}

    # Sample data
    for data, label in train_data_with_labels:
        if len(sampled_data_dict[label]) < min_class_count:
            sampled_data_dict[label].append(data)

    # Flatten the sampled data and labels
    for label, data_list in sampled_data_dict.items():
        sampled_train_data.extend(data_list)
        sampled_train_labels.extend([label] * len(data_list))

    return sampled_train_data, sampled_train_labels