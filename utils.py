import random
import re

import librosa
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


def spectral_distances(
    z: torch.Tensor, metric: str = "cosine", front_pad: bool = False, **kwargs
) -> np.ndarray:
    """
    Computes spectral distances between consecutive elements in a tensor.

    Args:
        z (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        metric (str, optional): Metric to use ('cosine' or 'bounded_euclidean'). Defaults to 'cosine'.
        front_pad (bool, optional): Whether to pad the front of the result. Defaults to False.
        **kwargs: Additional keyword arguments for the similarity metric.

    Returns:
        np.ndarray: Array of spectral distances.
    """
    z1 = z[:, :-1, :]
    z2 = z[:, 1:, :]
    if metric == "cosine":
        d = 1 - F.cosine_similarity(z1, z2, dim=-1)
    elif metric == "bounded_euclidean":
        d = 1 - bounded_euclidean_similarity(z1, z2, dim=-1, **kwargs)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    d = d.squeeze().cpu().numpy()
    if front_pad:
        d = np.insert(d, 0, d[0])
    return d


def box_convolve(y: np.ndarray, box_pts: int) -> np.ndarray:
    """
    Applies a box convolution (moving average) to smooth the input array.

    Args:
        y (np.ndarray): Input array.
        box_pts (int): Number of points in the box.

    Returns:
        np.ndarray: Smoothed array.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


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


def get_filename(wav: str) -> str:
    """
    Extracts the filename without extension from a WAV file path.

    Args:
        wav (str): Path to the WAV file.

    Returns:
        str: Filename without extension.
    """
    return re.split(r"[\\/]", wav)[-1][:-4]


def gen_spec(x: np.ndarray, n: int, h: int, c: bool = False) -> np.ndarray:
    """
    Generates a spectrogram from an audio signal.

    Args:
        x (np.ndarray): Audio signal.
        n (int): Number of FFT components.
        h (int): Hop length.
        c (bool, optional): Whether to center the signal. Defaults to False.

    Returns:
        np.ndarray: Spectrogram in decibels.
    """
    S = librosa.stft(x, n_fft=n, hop_length=h, center=c)
    D = librosa.amplitude_to_db(np.abs(S), ref=1)
    return D


def frame_to_time(
    frame: int, hop_length: int, sample_rate: int, shift: bool = False
) -> float:
    """
    Converts a frame index to time in seconds.

    Args:
        frame (int): Frame index.
        hop_length (int): Hop length in samples.
        sample_rate (int): Sample rate in Hz.
        shift (bool, optional): Whether to shift the frame by one. Defaults to False.

    Returns:
        float: Time in seconds.
    """
    if shift:
        frame += 1
    return frame * (hop_length / sample_rate)


def sample_to_time(sample: int, sample_rate: int) -> float:
    """
    Converts a sample index to time in seconds.

    Args:
        sample (int): Sample index.
        sample_rate (int): Sample rate in Hz.

    Returns:
        float: Time in seconds.
    """
    return sample / sample_rate


def coarsen_selections(
    onsets: np.ndarray, offsets: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarsens overlapping selections based on a threshold.

    Args:
        onsets (np.ndarray): Array of onset times.
        offsets (np.ndarray): Array of offset times.
        threshold (float): Threshold for coarsening in seconds.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Coarsened onsets and offsets.
    """
    deltas = onsets[1:] - offsets[:-1]
    offset_indices = np.where(deltas < threshold)[0]
    onset_indices = offset_indices + 1
    onsets = np.delete(onsets, onset_indices)
    offsets = np.delete(offsets, offset_indices)
    return onsets, offsets
