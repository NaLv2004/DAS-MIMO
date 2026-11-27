from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class DetectorResult:
    """Outputs shared by all reproduced learned detectors.

    ``trajectory`` contains one soft symbol estimate per unfolded layer.  Every
    tensor is real-valued and shaped ``[batch, dimension, 1]``.  For a complex
    system the real and imaginary parts occupy the first and second halves.
    ``aux`` contains model-specific tensors needed to evaluate the paper loss.
    """

    trajectory: list[torch.Tensor]
    aux: dict[str, object] = field(default_factory=dict)

    @property
    def final(self) -> torch.Tensor:
        if not self.trajectory:
            raise RuntimeError("detector returned an empty trajectory")
        return self.trajectory[-1]


def as_column(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3 and x.shape[-1] == 1:
        return x
    raise ValueError(f"expected [batch, n] or [batch, n, 1], got {tuple(x.shape)}")


def pam_levels(
    qam_order: int, *, normalize: bool, device=None, dtype=None
) -> torch.Tensor:
    """PAM alphabet for BPSK or square QAM, in ascending order."""

    order = int(qam_order)
    if order == 2:
        levels = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
        return levels
    root = math.isqrt(order)
    if root * root != order or root < 2:
        raise ValueError("qam_order must be 2 or a square QAM order")
    levels = torch.arange(-(root - 1), root, 2, device=device, dtype=dtype)
    if normalize:
        # Average complex-symbol energy is one, hence each real component has
        # average energy 1/2.
        levels = levels / math.sqrt(2.0 * (order - 1) / 3.0)
    return levels


def nearest_level(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    levels = levels.to(device=x.device, dtype=x.dtype)
    indices = (x.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return levels[indices]


def batched_trace(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def stable_solve(
    matrix: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Batched solve with a scale-aware diagonal jitter for rare ill conditioning."""

    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must be square")
    n = matrix.shape[-1]
    scale = batched_trace(matrix).abs().div(float(n)).clamp_min(1.0)
    eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    regularized = matrix + eps * scale[..., None, None] * eye
    return torch.linalg.solve(regularized, rhs)
