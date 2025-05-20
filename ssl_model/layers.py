from typing import Optional, Sequence, List, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as F_audio


class HighPassFilter(nn.Module):
    def __init__(
        self,
        cutoff_freq: float,
        sample_rate: int,
        Q: float = 0.7071,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time) or (batch, 1, time)
        returns same shape
        """
        need_unsqueeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            need_unsqueeze = True
        y = F_audio.highpass_biquad(
            x, self.sample_rate, self.cutoff_freq, self.Q
        )
        return y.squeeze(1) if need_unsqueeze else y


class MLP(nn.Module):
    """
    Simple multi‐layer perceptron:
      - layers = [hidden_dim] ⇒ one hidden + out
      - layers = []          ⇒ single linear
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        layers: Sequence[int] = (),
        dropout: float = 0,
        activation: nn.Module = nn.LeakyReLU(),
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        dims = [in_dim, *layers, out_dim]
        seq: List[nn.Module] = []
        for i in range(len(dims) - 1):
            seq.append(nn.Dropout(dropout))
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            # only add activation between hidden layers
            if i < len(dims) - 2:
                seq.append(activation)
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTransform(nn.Module):
    """
    Builds a stacked 1D-conv encoder from a list of (out_ch, kernel, stride, pad).
    If you only want one layer, give a single-entry list.
    """

    def __init__(
        self,
        in_channels: int = 1,
        layer_configs: Sequence[Tuple[int, int, int, int]] = ((64, 8, 4, 3),),
        use_bn: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        bias: bool = True,
    ):
        super().__init__()

        modules: List[nn.Module] = []
        ch = in_channels
        for idx, (out_ch, k, s, p) in enumerate(layer_configs):
            modules.append(
                nn.Conv1d(ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
            )
            # if use_bn:
            #     modules.append(nn.BatchNorm1d(out_ch, affine=bias))
            modules.append(activation)
            ch = out_ch

        self.transform = nn.Sequential(*modules)
        self.latent_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
