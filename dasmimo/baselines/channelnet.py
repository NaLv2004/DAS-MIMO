from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import DetectorResult


class _SharedMLP(nn.Module):
    """Apply one MLP independently, with shared weights, to every antenna."""

    def __init__(self, widths: list[int], *, final_linear: bool = False):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("an MLP needs at least an input and output width")
        self.layers = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1])
            for index in range(len(widths) - 1)
        )
        self.final_linear = bool(final_linear)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            features = layer(features)
            if index + 1 < len(self.layers) or not self.final_linear:
                features = F.relu(features)
        return features


class ChannelNetMLP(nn.Module):
    """Permutation-equivariant ChannelNet for i.i.d. Rayleigh channels.

    Inputs use the paper's real-equivalent system: ``H`` is ``[B,N,K]`` and
    ``y`` is ``[B,N,1]``.  The output contains one PAM-class logit vector per
    real transmit coordinate, shaped ``[B,K,M]``.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 10,
        iterations: int = 20,
        classes: int = 4,
        levels: torch.Tensor | None = None,
    ):
        super().__init__()
        if feature_dim <= 0 or iterations <= 0 or classes <= 1:
            raise ValueError("feature_dim/iterations/classes must be positive")
        self.feature_dim = int(feature_dim)
        self.iterations = int(iterations)
        self.classes = int(classes)
        if levels is None:
            raw = torch.arange(
                -(self.classes - 1), self.classes, 2, dtype=torch.float32
            )
            levels = raw / math.sqrt(2.0 * (self.classes**2 - 1) / 3.0)
        if levels.numel() != self.classes:
            raise ValueError("levels must contain one value per output class")
        self.register_buffer("levels", levels.detach().clone(), persistent=False)

        d = self.feature_dim
        self.receive_processors = nn.ModuleList(
            _SharedMLP([d, d, d]) for _ in range(self.iterations)
        )
        self.transmit_processors = nn.ModuleList(
            [
                *(_SharedMLP([d, d, d]) for _ in range(self.iterations - 1)),
                _SharedMLP([d, 5 * d, 5 * d, self.classes], final_linear=True),
            ]
        )

    def forward_logits(self, H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if H.dim() != 3 or y.dim() != 3 or y.shape[-1] != 1:
            raise ValueError("expected H=[B,N,K] and y=[B,N,1]")
        if H.shape[0] != y.shape[0] or H.shape[1] != y.shape[1]:
            raise ValueError("incompatible H and y dimensions")

        # Algorithm 1 writes F_rx <- y.  Since F_rx has d columns, this is the
        # unparameterized broadcast y 1_d^T depicted by the architecture.
        received = y.expand(-1, -1, self.feature_dim)
        previous_transmit_input = None
        logits = None
        Ht = H.transpose(1, 2)

        for index, (receive_mlp, transmit_mlp) in enumerate(
            zip(self.receive_processors, self.transmit_processors)
        ):
            received = receive_mlp(received)
            transmit_input = torch.bmm(Ht, received)
            if previous_transmit_input is not None:
                transmit_input = transmit_input + previous_transmit_input
            previous_transmit_input = transmit_input
            transmit = transmit_mlp(transmit_input)
            if index + 1 == self.iterations:
                logits = transmit
                break
            received = torch.bmm(H, transmit)
            received = received - y

        if logits is None:
            raise RuntimeError("ChannelNet produced no logits")
        return logits

    def forward(self, H: torch.Tensor, y: torch.Tensor) -> DetectorResult:
        logits = self.forward_logits(H, y)
        # The coded adapter consumes a continuous symbol estimate.  Returning
        # the posterior mean keeps the demodulator path differentiable; hard
        # argmax decisions are used only by uncoded SER evaluation.
        probabilities = torch.softmax(logits, dim=-1)
        return DetectorResult(
            [self.soft_symbols(logits, self.levels)],
            {"logits": logits, "probabilities": probabilities},
        )

    @staticmethod
    def labels(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        values = x.squeeze(-1)
        levels = levels.to(device=values.device, dtype=values.dtype)
        return (values.unsqueeze(-1) - levels).abs().argmin(dim=-1)

    def paper_loss(
        self,
        logits: torch.Tensor | DetectorResult,
        x: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(logits, DetectorResult):
            logits = logits.aux["logits"]
        labels = self.labels(x, levels)
        return F.cross_entropy(logits.reshape(-1, self.classes), labels.reshape(-1))

    def soft_symbols(self, logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.to(device=logits.device, dtype=logits.dtype)
        return (torch.softmax(logits, dim=-1) * levels).sum(dim=-1, keepdim=True)

    def detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        levels: torch.Tensor | None = None,
    ) -> DetectorResult:
        """Return logits for paper CE and PAM posterior means for decoding."""

        logits = self.forward_logits(H, y)
        levels = self.levels if levels is None else levels
        return DetectorResult(
            [self.soft_symbols(logits, levels)],
            {"logits": logits, "probabilities": torch.softmax(logits, dim=-1)},
        )

    def hard_symbols(self, logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.to(device=logits.device, dtype=logits.dtype)
        return levels[logits.argmax(dim=-1)].unsqueeze(-1)
