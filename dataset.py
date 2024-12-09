import random

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from preprocess import WatkinsSpermWhalePreprocess


class SpermWhaleClicks(Dataset):
    def __init__(
        self,
        n_samples: int,
        subset: str,
        base_path: str = "data/wavs/*.wav",
        window: float = 0.5,
        window_pad: int = 136,
        sample_rate: int = 48000,
        seed: int = 42,
        epsilon: float = 2e-6,
    ) -> None:
        """
        Initializes the dataset for Sperm Whale Clicks.

        Args:
            n_samples (int): Number of samples in the dataset.
            subset (str): Subset type ('train', 'val', 'test').
            base_path (str): Glob pattern to locate WAV files.
            window (float): Window duration in seconds.
            window_pad (int): Padding for the window.
            sample_rate (int): Sampling rate for audio.
            seed (int): Random seed for reproducibility.
            epsilon (float): Small value to prevent boundary issues.
        """
        self.n_samples = n_samples
        self.subset = subset
        self.window = window
        self.window_pad = window_pad
        self.sample_rate = sample_rate
        self.seed = seed
        self.epsilon = epsilon

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.preprocesser = WatkinsSpermWhalePreprocess(
            base_path=base_path,
            window_width=self.window + self.window_pad / self.sample_rate,
        )
        self.wavs, self.weights = self.preprocesser.run()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.subset != "train":
            np.random.seed(idx)
            random.seed(idx)

        wav = np.random.choice(self.wavs, p=self.weights)
        dur = sf.info(wav).duration

        max_start = dur - (
            self.window + self.window_pad / self.sample_rate + self.epsilon
        )
        if max_start <= 0:
            start_time = 0.0
        else:
            start_time = random.uniform(0, max_start)

        x, _ = librosa.load(
            wav,
            offset=start_time,
            duration=self.window + self.window_pad / self.sample_rate + self.epsilon,
            sr=self.sample_rate,
        )
        window_frames = int(self.window * self.sample_rate) + self.window_pad
        x = librosa.util.fix_length(data=x, size=window_frames)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0)
        return x, x
