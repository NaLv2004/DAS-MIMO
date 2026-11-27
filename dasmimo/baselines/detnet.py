from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import DetectorResult, as_column


class _ComplexQAMDetNetLayer(nn.Module):
    def __init__(
        self,
        real_transmit_dim: int,
        hidden_dim: int,
        state_dim: int,
        complex_symbols: int,
        qam_order: int,
    ):
        super().__init__()
        self.delta1 = nn.Parameter(torch.zeros(()))
        self.delta2 = nn.Parameter(torch.zeros(()))
        self.W1 = nn.Linear(real_transmit_dim + state_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, complex_symbols * qam_order)
        self.W3 = nn.Linear(hidden_dim, state_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.W1, self.W2, self.W3):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.normal_(module.bias, std=0.01)
        with torch.no_grad():
            self.delta1.zero_()
            self.delta2.zero_()


class ComplexQAMDetNet(nn.Module):
    """Released hard-output complex-QAM DetNet architecture.

    This is the direct real-equivalent generalization used by the authors in
    ``DetNet_HD_16QAM.py``: one joint 16-class output per complex transmit
    symbol, ``L=30`` with 29 executed updates, hidden width ``8*(2K)``, and
    auxiliary-state width ``4*(2K)``.  The supplied PAM levels may be
    normalized; this preserves the repository's unit-energy 16-QAM convention.
    """

    def __init__(
        self,
        real_transmit_dim: int,
        levels: torch.Tensor,
        *,
        num_layers: int = 30,
        hidden_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        residual_alpha: float = 0.9,
    ):
        super().__init__()
        self.real_transmit_dim = int(real_transmit_dim)
        if self.real_transmit_dim % 2:
            raise ValueError("complex-QAM DetNet requires an even real dimension")
        self.complex_symbols = self.real_transmit_dim // 2
        self.register_buffer("levels", torch.as_tensor(levels).clone())
        root = int(self.levels.numel())
        self.qam_order = root * root
        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("source layer count must be at least two")
        self.hidden_dim = int(hidden_dim or 8 * self.real_transmit_dim)
        self.state_dim = int(state_dim or 4 * self.real_transmit_dim)
        self.residual_alpha = float(residual_alpha)
        real = self.levels.repeat_interleave(root)
        imag = self.levels.repeat(root)
        self.register_buffer("constellation_real", real)
        self.register_buffer("constellation_imag", imag)
        self.layers = nn.ModuleList(
            _ComplexQAMDetNetLayer(
                self.real_transmit_dim,
                self.hidden_dim,
                self.state_dim,
                self.complex_symbols,
                self.qam_order,
            )
            for _ in range(self.num_layers - 1)
        )

    def _one_hot(self, x: torch.Tensor) -> torch.Tensor:
        x = as_column(x).squeeze(-1)
        k = self.complex_symbols
        distance = (x[:, :k, None] - self.constellation_real) ** 2 + (
            x[:, k:, None] - self.constellation_imag
        ) ** 2
        return F.one_hot(distance.argmin(dim=-1), num_classes=self.qam_order).to(
            x.dtype
        )

    def _logits_to_real(self, logits: torch.Tensor) -> torch.Tensor:
        real = (logits * self.constellation_real).sum(dim=-1)
        imag = (logits * self.constellation_imag).sum(dim=-1)
        return torch.cat([real, imag], dim=1).unsqueeze(-1)

    def hard_decision(self, result: DetectorResult) -> torch.Tensor:
        """Apply the released graph's joint-QAM argmax decision.

        The 16 outputs are unconstrained regression scores, not probabilities.
        Consequently, slicing ``result.final`` to the nearest PAM point is not
        equivalent to the author's ``argmax(S3, axis=2)`` SER calculation.
        """

        logits = result.aux["joint_qam_logits_trajectory"][-1]
        indices = logits.argmax(dim=-1)
        real = self.constellation_real[indices]
        imag = self.constellation_imag[indices]
        return torch.cat([real, imag], dim=1).unsqueeze(-1)

    def forward(self, H: torch.Tensor, y: torch.Tensor, **_) -> DetectorResult:
        y = as_column(y)
        ht = H.transpose(1, 2)
        hty, hth = torch.bmm(ht, y), torch.bmm(ht, H)
        x = H.new_zeros(H.shape[0], self.real_transmit_dim, 1)
        state = H.new_zeros(H.shape[0], self.state_dim)
        previous_logits = H.new_zeros(H.shape[0], self.complex_symbols, self.qam_order)
        estimates, logits_trajectory, state_trajectory = [], [], []
        alpha = self.residual_alpha
        for layer in self.layers:
            q = x - layer.delta1 * hty + layer.delta2 * torch.bmm(hth, x)
            hidden = F.relu(layer.W1(torch.cat([q.squeeze(-1), state], dim=1)))
            new_logits = layer.W2(hidden).view(
                H.shape[0], self.complex_symbols, self.qam_order
            )
            new_state = layer.W3(hidden)
            logits = (1.0 - alpha) * new_logits + alpha * previous_logits
            state = (1.0 - alpha) * new_state + alpha * state
            x = self._logits_to_real(logits)
            estimates.append(x)
            logits_trajectory.append(logits)
            state_trajectory.append(state)
            previous_logits = logits
        return DetectorResult(
            estimates,
            {
                "joint_qam_logits_trajectory": logits_trajectory,
                "state_trajectory": state_trajectory,
            },
        )

    def paper_loss(self, result: DetectorResult, x_true: torch.Tensor) -> torch.Tensor:
        target = self._one_hot(x_true)
        trajectory = result.aux["joint_qam_logits_trajectory"]
        total = target.new_zeros(())
        for index, logits in enumerate(trajectory, start=1):
            total = (
                total
                + math.log(index) * (target - logits).square().flatten(1).mean(1).mean()
            )
        return total
