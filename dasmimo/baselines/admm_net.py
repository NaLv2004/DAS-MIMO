from __future__ import annotations

import torch
from torch import nn

from .common import DetectorResult, as_column, nearest_level, pam_levels, stable_solve


def smooth_constellation_projection(
    x: torch.Tensor,
    levels: torch.Tensor,
    theta: torch.Tensor | float,
) -> torch.Tensor:
    """Equation (22): sum_i tanh(theta * (x - tau_i))."""

    levels = levels.to(x)
    thresholds = 0.5 * (levels[:-1] + levels[1:])
    theta = torch.as_tensor(theta, device=x.device, dtype=x.dtype)
    while theta.dim() < x.dim():
        theta = theta.unsqueeze(-1)
    half_step = 0.5 * (levels[1] - levels[0])
    center = 0.5 * (levels[0] + levels[-1])
    return center + half_step * torch.tanh(
        theta.unsqueeze(-1) * (x.unsqueeze(-1) - thresholds)
    ).sum(dim=-1)


class ADMMNet(nn.Module):
    """Algorithm 1 with five learned scalars per unfolded layer."""

    def __init__(
        self,
        transmit_dim: int,
        receive_dim: int,
        levels: torch.Tensor,
        *,
        num_layers: int = 30,
        hard_projection_theta: float = 100.0,
    ):
        super().__init__()
        self.transmit_dim = int(transmit_dim)
        self.receive_dim = int(receive_dim)
        self.num_layers = int(num_layers)
        self.hard_projection_theta = float(hard_projection_theta)
        self.register_buffer("levels", torch.as_tensor(levels).clone())
        self.delta = nn.Parameter(torch.full((self.num_layers,), 1e-3))
        self.theta = nn.Parameter(torch.full((self.num_layers,), 1.5))
        self.alpha = nn.Parameter(torch.full((self.num_layers,), 1.5))
        self.beta = nn.Parameter(torch.full((self.num_layers,), 0.1))
        gamma = torch.arange(self.num_layers, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma / (gamma + 5.0))

    @classmethod
    def for_qam(
        cls,
        n_tx: int,
        n_rx: int,
        qam_order: int,
        *,
        normalize: bool = False,
        **kwargs,
    ) -> "ADMMNet":
        complex_system = int(qam_order) != 2
        factor = 2 if complex_system else 1
        return cls(
            factor * int(n_tx),
            factor * int(n_rx),
            pam_levels(qam_order, normalize=normalize),
            **kwargs,
        )

    def _initial_estimate(self, H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Ht = H.transpose(1, 2)
        return stable_solve(torch.bmm(Ht, H), torch.bmm(Ht, y))

    def forward(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        initial_estimate: torch.Tensor | None = None,
        **_,
    ) -> DetectorResult:
        y = as_column(y)
        if H.shape[1:] != (self.receive_dim, self.transmit_dim):
            raise ValueError(
                f"expected H[..., {self.receive_dim}, {self.transmit_dim}], got {tuple(H.shape)}"
            )
        Ht = H.transpose(1, 2)
        x = (
            self._initial_estimate(H, y)
            if initial_estimate is None
            else as_column(initial_estimate)
        )
        if x.shape != (H.shape[0], self.transmit_dim, 1):
            raise ValueError("initial_estimate has an incompatible shape")
        e = H.new_zeros(H.shape[0], self.receive_dim, 1)
        v_hat = e.clone()
        v_previous = e.clone()
        trajectory: list[torch.Tensor] = []
        hard_trajectory: list[torch.Tensor] = []
        residual_trajectory: list[torch.Tensor] = []
        accelerated_trajectory: list[torch.Tensor] = []

        for layer in range(self.num_layers):
            corrected = e + (1.0 - 2.0 * self.beta[layer]) * v_hat
            x = x - self.delta[layer] * torch.bmm(Ht, corrected)
            x = smooth_constellation_projection(x, self.levels, self.theta[layer])
            e = torch.bmm(H, x) - y
            v = (
                self.alpha[layer] * e
                + (1.0 - self.alpha[layer] * self.beta[layer]) * v_hat
            )
            v_hat = v + self.gamma[layer] * (v - v_previous)
            v_previous = v
            hard_soft = smooth_constellation_projection(
                x, self.levels, self.hard_projection_theta
            )
            trajectory.append(x)
            hard_trajectory.append(hard_soft)
            residual_trajectory.append(e)
            accelerated_trajectory.append(v_hat)

        return DetectorResult(
            trajectory,
            {
                "hard_soft_trajectory": hard_trajectory,
                "residual_trajectory": residual_trajectory,
                "accelerated_trajectory": accelerated_trajectory,
            },
        )

    def paper_loss(self, result: DetectorResult, x_true: torch.Tensor) -> torch.Tensor:
        """Equation (29): sum of layer-wise squared symbol errors."""

        x_true = as_column(x_true)
        hard_trajectory = result.aux.get("hard_soft_trajectory")
        if not isinstance(hard_trajectory, list) or not hard_trajectory:
            raise ValueError("ADMM-Net result is missing hard-soft layer outputs")
        return sum(
            (estimate - x_true).square().sum(dim=1).mean()
            for estimate in hard_trajectory
        )

    @torch.no_grad()
    def hard_decision(self, result: DetectorResult) -> torch.Tensor:
        return nearest_level(result.final, self.levels)
