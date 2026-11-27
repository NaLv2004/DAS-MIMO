import math

import torch


class SpatialChannelSampler:
    def __init__(self, config):
        self.shape = (config.Nr, config.Nt)
        self.device = config.DEVICE
        self.dtype = config.dtype

    def sample_complex(self, batch_size):
        real = torch.randn(
            batch_size, *self.shape, device=self.device, dtype=self.dtype
        )
        imag = torch.randn_like(real)
        return torch.complex(real, imag) * math.sqrt(0.5)
