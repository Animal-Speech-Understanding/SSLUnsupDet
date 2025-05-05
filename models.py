import torch
from torch import nn

from layers import MLP, HighPassFilter, ConvTransform


class SpectralBoundaryEncoder(nn.Module):
    def __init__(
        self,
        filter_params: dict[str, any],
        conv_transform_params: dict[str, any],
        mlp_params: dict[str, any],
    ):
        super(SpectralBoundaryEncoder, self).__init__()

        self.filter_params = filter_params
        self.conv_transform_params = conv_transform_params
        self.mlp_params = mlp_params

        self.preprocess = HighPassFilter(**self.filter_params)

        self.transform = ConvTransform(**self.conv_transform_params)

        self.postprocess = nn.Sequential(
            MLP(
                in_dim=self.transform.latent_dim,
                **self.mlp_params,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.transform(x).transpose(1, 2)
        z = self.postprocess(x)
        return z
