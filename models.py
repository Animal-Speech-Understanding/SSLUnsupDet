import numpy as np
import torch
import torch.nn as nn

from layers import MLP, STFT, ConvTransform, HighPassFilter, IdentityLayer, LambdaLayer


class SpectralBoundaryEncoder(nn.Module):
    def __init__(self, 
                 preprocess_params: dict[str, object],
                 transform_params: dict[str, object],
                 postprocess_params: dict[str, object]):
        super(SpectralBoundaryEncoder, self).__init__()
        
        self.preprocess_params = preprocess_params
        self.transform_params = transform_params
        self.postprocess_params = postprocess_params
        
        # Preprocessing Layer
        try:
            filter_params = self.preprocess_params['filter_params']
            cutoff_freq = filter_params['cutoff_freq']
            b = filter_params['b']
            sr = filter_params['sample_rate']
            ramp_duration = filter_params['ramp_duration']
            self.preprocess = HighPassFilter(
                cutoff_freq=cutoff_freq,
                sample_rate=sr,
                b=b,
                ramp_duration=ramp_duration
            )
        except KeyError:
            self.preprocess = IdentityLayer()

        # Transformation Layer
        if self.transform_params['name'] == 'stft':
            params = self.transform_params['params']
            self.transform = STFT(**params)
            lambd = self._get_postprocess_lambda()
            self.postprocess = LambdaLayer(lambd=lambd)
        elif 'conv1d' in self.transform_params['name']:
            params = self.transform_params['params']
            self.transform = ConvTransform(**params)
            lambd, mlp = self._get_postprocess_components()
            if mlp is not None:
                self.postprocess = nn.Sequential(
                    LambdaLayer(lambd=lambd),
                    mlp,
                    self._get_output_activation()
                )
            else:
                self.postprocess = nn.Sequential(
                    LambdaLayer(lambd=lambd),
                    self._get_output_activation()
                )
        else:
            raise ValueError(f"Unsupported transform name: {self.transform_params['name']}")

    def _get_postprocess_lambda(self):
        lambd_type = self.postprocess_params.get('lambd', 'concat')
        if lambd_type == 'concat':
            return lambda x: torch.cat(x, dim=1).transpose(1, 2)
        elif lambd_type == 0:
            return lambda x: x[0].transpose(1, 2)
        elif lambd_type == 1:
            return lambda x: x[1].transpose(1, 2)
        else:
            raise ValueError(f"Unsupported lambd type: {lambd_type}")

    def _get_postprocess_components(self):
        mlp_params = self.postprocess_params.get('mlp_params', None)
        def lambd(x):
            return x.transpose(1, 2)
        in_dimension = self.transform.latent_dim  # Ensure ConvTransform defines latent_dim
        
        if mlp_params is not None:
            mlp = MLP(in_dimension=in_dimension, **mlp_params)
            return lambd, mlp
        else:
            return lambd, None

    def _get_output_activation(self):
        activation = self.postprocess_params.get('output_activation', None)
        if activation is None:
            return IdentityLayer()
        activation_map = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'hardtanh': nn.Hardtanh(),
        }
        if activation.lower() in activation_map:
            return activation_map[activation.lower()]
        else:
            raise ValueError(f"Unsupported output activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.transform(x)
        z = self.postprocess(x)
        return z
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(SpectralBoundaryEncoder, self).__str__() + f'\nTrainable parameters: {params}'
