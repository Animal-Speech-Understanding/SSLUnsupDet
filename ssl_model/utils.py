import random

import numpy as np
import torch
import torch.nn.functional as F


def bounded_euclidean_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Computes bounded Euclidean similarity between two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
        alpha (float, optional): Scaling factor. Defaults to 1.0.
        beta (float, optional): Offset. Defaults to 0.0.
        dim (int, optional): Dimension to compute similarity on. Defaults to -1.

    Returns:
        torch.Tensor: Similarity scores.
    """
    delta = torch.pow(x - y, 2)
    d = torch.sum(delta, dim=dim)
    return 1 - torch.tanh(F.relu(alpha * d + beta))


def seed_everything(seed: int) -> None:
    """
    Sets seeds for reproducibility across various libraries.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_similarity_fn(sim_metric_params: dict):
    name = sim_metric_params.get('name', 'cosine')
    if name == 'cosine':
        return lambda z1, z2: F.cosine_similarity(z1, z2, dim=-1)
    elif name == 'bounded_euclidean':
        params = sim_metric_params.get('params', {})
        return lambda z1, z2: bounded_euclidean_similarity(z1, z2, dim=-1, **params)
    else:
        raise ValueError(f"Unknown sim_metric_params.name: {name}")
