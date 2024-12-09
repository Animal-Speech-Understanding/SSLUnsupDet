import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPassFilter(nn.Module):
    """
    Implements a High-Pass Filter using a sinc function and convolution.
    """

    def __init__(
        self,
        cutoff_freq: float,
        sample_rate: int,
        b: float = 0.08,
        eps: float = 1e-20,
        ramp_duration: float | None = None,
    ) -> None:
        """
        Initializes the HighPassFilter.

        Args:
            cutoff_freq (float): Cutoff frequency of the high-pass filter in Hz.
            sample_rate (int): Sampling rate of the input signal in Hz.
            b (float, optional): Transition bandwidth. Defaults to 0.08.
            eps (float, optional): Small epsilon value to prevent division by zero. Defaults to 1e-20.
            ramp_duration (Optional[float], optional): Duration of the Hann ramp in seconds. If None, no ramp is applied. Defaults to None.
        """
        super(HighPassFilter, self).__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.fc = cutoff_freq / sample_rate
        self.b = b

        # Calculate filter length N based on transition bandwidth
        N = int(np.ceil(4 / b))
        N += 1 - N % 2  # Ensure N is odd
        self.N = N

        # Initialize parameters
        self.epsilon = nn.Parameter(torch.tensor(eps), requires_grad=False)
        self.window = nn.Parameter(torch.blackman_window(self.N), requires_grad=False)

        # Create the sinc filter
        n = torch.arange(self.N)
        self.sinc_fx = nn.Parameter(
            self.sinc(2 * self.fc * (n - (self.N - 1) / 2.0)), requires_grad=False
        )

        self.ramp_duration = ramp_duration
        if self.ramp_duration is not None:
            self.ramp = nn.Parameter(
                self.hann_ramp(self.sample_rate, self.ramp_duration),
                requires_grad=False,
            )
        else:
            self.ramp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the high-pass filter to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, signal_length).

        Returns:
            torch.Tensor: Filtered output tensor of shape (batch_size, signal_length).
        """
        original_size = x.size()
        x = x.view(
            x.size(0), 1, x.size(-1)
        )  # Reshape to (batch_size, 1, signal_length)

        # Apply window to the sinc filter
        sinc_fx = self.sinc_fx * self.window
        sinc_fx = sinc_fx / torch.sum(sinc_fx)
        sinc_fx = -sinc_fx
        sinc_fx[int((self.N - 1) / 2)] += 1  # Add delta function

        # Perform convolution
        output = F.conv1d(x, sinc_fx.view(-1, 1, self.N), padding=self.N // 2)

        # Apply ramp if specified
        if self.ramp is not None:
            ramp_length = len(self.ramp)
            output[:, :, :ramp_length] *= torch.flip(self.ramp, dims=[0])
            output[:, :, -ramp_length:] *= self.ramp

        return output.view(original_size)  # Reshape back to original size

    def sinc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the sinc function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying sinc.
        """
        y = math.pi * torch.where(x == 0, self.epsilon, x)
        return torch.sin(y) / y

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the HighPassFilter.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return {
            "name": "HighPassFilter",
            "cutoff_freq": self.cutoff_freq,
            "sample_rate": self.sample_rate,
            "b": self.b,
        }

    @staticmethod
    def hann_ramp(sample_rate: int, ramp_duration: float = 0.002) -> torch.Tensor:
        """
        Generates a Hann ramp.

        Args:
            sample_rate (int): Sampling rate in Hz.
            ramp_duration (float, optional): Duration of the ramp in seconds. Defaults to 0.002.

        Returns:
            torch.Tensor: Hann ramp tensor.
        """
        t = torch.arange(0, ramp_duration, 1 / sample_rate)
        ramp = 0.5 * (1.0 + torch.cos((math.pi / ramp_duration) * t))
        return ramp


class STFT(nn.Module):
    """
    Computes the Short-Time Fourier Transform (STFT) of the input signal.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        coords: str = "polar",
        dB: bool = False,
        center: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        """
        Initializes the STFT layer.

        Args:
            kernel_size (int): Number of FFT components.
            stride (int): Hop length between successive frames.
            coords (str, optional): Coordinate system for output ('cartesian' or 'polar'). Defaults to 'polar'.
            dB (bool, optional): Whether to convert magnitude to decibels. Defaults to False.
            center (bool, optional): Whether to pad the signal to center the FFT window. Defaults to True.
            epsilon (float, optional): Small value to prevent log of zero. Defaults to 1e-8.
        """
        super(STFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.coords = coords.lower()
        self.dB = dB
        self.center = center
        self.epsilon = epsilon

        self.register_buffer("window", torch.hann_window(self.kernel_size))

        if self.dB and self.coords != "polar":
            raise ValueError("dB scaling requires 'polar' coordinates.")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        Applies STFT to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, signal_length).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
                - If coords='cartesian': (S_real, S_imag)
                - If coords='polar': (S_phase, S_mag) or (S_phase, S_mag_db) if dB=True
        """
        # Compute STFT
        S = torch.stft(
            x.squeeze(dim=1),  # Shape: (batch_size, signal_length)
            n_fft=self.kernel_size,
            hop_length=self.stride,
            window=self.window,
            onesided=True,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            return_complex=False,
        )
        S_real = S[..., 0]  # Shape: (batch_size, n_freq, n_frames)
        S_imag = S[..., 1]

        if self.coords == "cartesian":
            return S_real, S_imag
        elif self.coords == "polar":
            # Compute magnitude and phase
            S_mag = torch.sqrt(S_real**2 + S_imag**2) + self.epsilon
            S_phase = torch.atan2(S_imag, S_real)
            if self.dB:
                S_mag = self.amplitude_to_db(S_mag)
            return S_phase, S_mag
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coords}")

    def get_out_size(self, in_size: tuple[int, int, int]) -> tuple[int, int, int]:
        """
        Computes the output size after applying STFT.

        Args:
            in_size (Tuple[int, int, int]): Input size as (batch, channels, signal_length).

        Returns:
            Tuple[int, int, int]: Output size as (batch, n_freq, n_frames).
        """
        batch, in_channels, signal_length = in_size
        if self.center:
            signal_length += self.kernel_size // 2
        n_freq = self.kernel_size // 2 + 1
        n_frames = 1 + (signal_length - self.kernel_size) // self.stride
        return (batch, n_freq, n_frames)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the STFT layer.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return {
            "name": "STFT",
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dB_scaling": self.dB,
        }

    @staticmethod
    def amplitude_to_db(
        S: torch.Tensor, amin: float = 1e-5, delta_db: float | None = 80.0
    ) -> torch.Tensor:
        """
        Converts amplitude spectrogram to decibel scale.

        Args:
            S (torch.Tensor): Magnitude spectrogram.
            amin (float, optional): Minimum amplitude. Defaults to 1e-5.
            delta_db (Optional[float], optional): Dynamic range for dB scaling. If None, no clipping is applied. Defaults to 80.0.

        Returns:
            torch.Tensor: Spectrogram in decibels.
        """
        S = torch.clamp(S, min=amin)
        D = 20.0 * torch.log10(S)
        if delta_db is not None:
            D = torch.clamp(D, min=D.max() - delta_db)
        return D


class LambdaLayer(nn.Module):
    """
    Implements a layer that applies a given lambda function.
    """

    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initializes the LambdaLayer.

        Args:
            lambd (Callable[[torch.Tensor], torch.Tensor]): Lambda function to apply.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the lambda function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the lambda function.
        """
        return self.lambd(x)


class IdentityLayer(nn.Module):
    """
    Implements an identity layer that returns the input as-is.
    """

    def __init__(self) -> None:
        """
        Initializes the IdentityLayer.
        """
        super(IdentityLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the input tensor unchanged.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Unchanged input tensor.
        """
        return x


class MLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron (MLP) with optional linear projection.
    """

    def __init__(
        self,
        in_dimension: int,
        out_dimension: int | None = None,
        dropout: float = 0.5,
        linear_projection: bool = True,
    ) -> None:
        """
        Initializes the MLP.

        Args:
            in_dimension (int): Input dimension size.
            out_dimension (Optional[int], optional): Output dimension size. If None, defaults to in_dimension. Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            linear_projection (bool, optional): Whether to use a single linear projection. If False, uses two linear layers with activation. Defaults to True.
        """
        super(MLP, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = (
            out_dimension if out_dimension is not None else in_dimension
        )
        self.dropout = dropout
        self.linear_projection = linear_projection

        if self.linear_projection:
            layers = [
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_dimension, self.out_dimension),
            ]
        else:
            layers = [
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_dimension, self.in_dimension),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_dimension, self.out_dimension),
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dimension).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dimension).
        """
        return self.net(x)

    def __str__(self) -> str:
        """
        Returns a string representation of the MLP, including the number of trainable parameters.

        Returns:
            str: String representation.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f"{super(MLP, self).__str__()}\nTrainable parameters: {params}"


class ConvTransform(nn.Module):
    """
    Implements a convolutional transformation with various encoder options.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        encoder_name: str = "conv1d",
        out_channels: int | None = None,
        center: bool = False,
        double_channels: bool = False,
        layers: list[list[int]] | None = None,
        bias: bool = False,
        affine: bool = False,
    ) -> None:
        """
        Initializes the ConvTransform.

        Args:
            kernel_size (int): Kernel size for convolution.
            stride (int): Stride for convolution.
            encoder_name (str, optional): Type of encoder ('conv1d', 'stft_db', 'conv1d_layers'). Defaults to 'conv1d'.
            out_channels (Optional[int], optional): Number of output channels. If None, defaults to kernel_size // 2 + 1. Defaults to None.
            center (bool, optional): Whether to apply padding to center the convolution. Defaults to False.
            double_channels (bool, optional): Whether to double the number of channels. Defaults to False.
            layers (Optional[List[List[int]]], optional): Additional convolutional layers configurations. Defaults to None.
            bias (bool, optional): Whether to include bias in convolutional layers. Defaults to False.
            affine (bool, optional): Whether batch normalization layers are affine. Defaults to False.
        """
        super(ConvTransform, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.encoder_name = encoder_name.lower()
        self.out_channels = (
            out_channels if out_channels is not None else (self.kernel_size // 2 + 1)
        )
        self.center = center
        self.double_channels = double_channels
        self.layers = layers
        self.bias = bias
        self.affine = affine

        if self.center:
            self.padding_params = {
                "padding": self.kernel_size // 2,
                "padding_mode": "reflect",
            }
        else:
            self.padding_params = {"padding": 0}

        if self.double_channels:
            self.out_channels *= 2

        if self.encoder_name == "conv1d":
            self.transform = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    bias=self.bias,
                    **self.padding_params,
                ),
                nn.BatchNorm1d(self.out_channels, affine=self.affine),
                nn.LeakyReLU(),
            )
        elif self.encoder_name == "stft_db":
            self.transform = nn.Sequential(
                STFT(
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dB=True,
                    coords="polar",
                    center=self.center,
                ),
                LambdaLayer(lambd=lambda x: x[1]),  # Extract magnitude (dB)
            )
        elif self.encoder_name == "conv1d_layers":
            conv1d_layers = [[8, 4, 0], [6, 3, 0], [4, 2, 0], [4, 2, 0]]
            transform_layers = [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.out_channels,
                    kernel_size=conv1d_layers[0][0],
                    stride=conv1d_layers[0][1],
                    padding=conv1d_layers[0][2],
                    bias=self.bias,
                ),
                nn.BatchNorm1d(self.out_channels, affine=self.affine),
                nn.LeakyReLU(),
            ]
            for layer_cfg in conv1d_layers[1:]:
                k, s, p = layer_cfg
                transform_layers.extend(
                    [
                        nn.Conv1d(
                            in_channels=self.out_channels,
                            out_channels=self.out_channels,
                            kernel_size=k,
                            stride=s,
                            padding=p,
                            bias=self.bias,
                        ),
                        nn.BatchNorm1d(self.out_channels, affine=self.affine),
                        nn.LeakyReLU(),
                    ]
                )
            self.transform = nn.Sequential(*transform_layers)
        else:
            raise ValueError(f"Unsupported encoder name: {self.encoder_name}")

        self.latent_dim = self.out_channels

        if self.layers is not None:
            in_channels = self.out_channels
            for i, layer_cfg in enumerate(self.layers):
                if len(layer_cfg) == 4:
                    out_channels, kernel, stride, padding = layer_cfg
                elif len(layer_cfg) == 3:
                    kernel, stride, padding = layer_cfg
                    out_channels = in_channels
                else:
                    raise ValueError(
                        "Each layer configuration must have 3 or 4 elements."
                    )

                assert stride == 1, "Only stride=1 is supported in additional layers."
                assert (
                    padding == (kernel - 1) // 2
                ), "Padding must be (kernel_size - 1) // 2."

                self.transform.add_module(
                    f"conv_{i + 1}",
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=self.bias,
                    ),
                )
                self.transform.add_module(
                    f"batchnorm_{i + 1}",
                    nn.BatchNorm1d(out_channels, affine=self.affine),
                )
                self.transform.add_module(f"leakyrelu_{i + 1}", nn.LeakyReLU())
                in_channels = out_channels
                self.latent_dim = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the convolutional transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, signal_length).

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.transform(x)

    def get_out_size(self, in_size: tuple[int, int, int]) -> tuple[int, int, int]:
        """
        Computes the output size after applying the convolutional transformation.

        Args:
            in_size (Tuple[int, int, int]): Input size as (batch, channels, signal_length).

        Returns:
            Tuple[int, int, int]: Output size as (batch, out_channels, output_length).
        """
        batch, in_channels, signal_length = in_size
        if self.encoder_name in ["conv1d", "stft_db"]:
            if self.center:
                output_length = (
                    signal_length
                    + self.padding_params["padding"] * 2
                    - self.kernel_size
                ) // self.stride + 1
            else:
                output_length = (signal_length - self.kernel_size) // self.stride + 1
            out_channels = self.out_channels
        else:
            # For other encoders, infer by passing a dummy tensor
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, signal_length)
                dummy_output = self.forward(dummy_input)
                out_channels, output_length = (
                    dummy_output.shape[1],
                    dummy_output.shape[2],
                )
        return (batch, out_channels, output_length)

    def __str__(self) -> str:
        """
        Returns a string representation of the ConvTransform, including the number of trainable parameters.

        Returns:
            str: String representation.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f"{super(ConvTransform, self).__str__()}\nTrainable parameters: {params}"
