from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass
class WatkinsSpermWhalePreprocess:
    """
    Scan a directory for .wav files and collect metadata (sample rates and durations).
    """
    base_path: str = "data/wavs/*.wav"
    window_width: float = 0.5
    wav_files: List[Path] = field(init=False)

    def __post_init__(self):
        self.wav_files = sorted(Path().glob(self.base_path))

    def get_valid_files(self) -> Tuple[List[Path], List[float], List[int]]:
        paths, durations, srs = [], [], []
        for wav in self.wav_files:
            info = sf.info(str(wav))
            if info.duration > self.window_width:
                paths.append(wav)
                durations.append(info.duration)
                srs.append(info.samplerate)
        return paths, durations, srs

    @staticmethod
    def compute_weights(durations: List[float]) -> np.ndarray:
        arr = np.array(durations, dtype=np.float32)
        return arr / arr.sum()

    def run(self) -> Tuple[List[Path], np.ndarray, List[float]]:
        paths, durations, _ = self.get_valid_files()
        weights = WatkinsSpermWhalePreprocess.compute_weights(durations)
        return paths, weights, durations


class SpermWhaleClicksDataset(Dataset):
    """
    SSL dataset sampling multiple random windows from sperm-whale-click recordings.
    Pass `n_samples` to control how many windows in an epoch.
    Uses duration-weighted sampling of files.
    """
    def __init__(
        self,
        n_samples: int,
        base_path: str = "data/wavs/*.wav",
        window_sec: float = 0.5,
        pad_frames: int = 136,
        sample_rate: int = 48_000,
        subset: str = "train",
        seed: int = 42,
        epsilon: float = 1e-6,
    ) -> None:
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.pad_frames = pad_frames
        self.total_frames = int(window_sec * sample_rate) + pad_frames
        self.epsilon = epsilon
        self.subset = subset.lower()
        self.base_seed = seed

        # Preprocess file list and weights
        preprocess = WatkinsSpermWhalePreprocess(
            base_path=base_path,
            window_width=window_sec + pad_frames / sample_rate,
        )
        self.wavs, self.weights, self.durations = preprocess.run()
        if not self.wavs:
            raise RuntimeError(f"No valid .wav files found in {base_path}")

        # Create sampler for training
        self.sampler: Optional[WeightedRandomSampler] = None
        if self.subset == "train":
            self.sampler = WeightedRandomSampler(
                weights=self.weights.tolist(),
                num_samples=self.n_samples,
                replacement=True,
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine file index
        if self.subset == "train" and self.sampler is not None:
            file_idx = idx  # idx comes from sampler
        else:
            file_idx = idx % len(self.wavs)

        wav_path = self.wavs[file_idx]
        duration = self.durations[file_idx]

        # Random offset within valid range
        max_offset = duration - (self.total_frames / self.sample_rate) - self.epsilon
        if max_offset <= 0:
            offset = 0.0
        else:
            gen = torch.Generator()
            if self.subset != "train":
                gen.manual_seed(self.base_seed + idx)
            offset = torch.rand(1, generator=gen).item() * max_offset

        # Load audio segment
        waveform, sr = librosa.load(
            str(wav_path),
            sr=self.sample_rate,
            offset=offset,
            duration=self.total_frames / self.sample_rate + self.epsilon,
        )
        waveform = librosa.util.fix_length(waveform, size=self.total_frames)

        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        return tensor, tensor

# Usage:
# dataset = SpermWhaleClicksDataset(n_samples=10000, subset="train", base_path="data/wavs/*.wav")
# loader = DataLoader(
#     dataset,
#     batch_size=32,
#     sampler=dataset.sampler,
#     num_workers=4,
# )
