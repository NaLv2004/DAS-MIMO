from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import LDPC5GParams, torch_edges


class OffsetMinSumDecoder(nn.Module):
    """Flooding offset-min-sum decoding with a fixed number of iterations."""

    def __init__(
        self,
        params: LDPC5GParams,
        num_iterations: int = 8,
        offset: float = 0.25,
        smooth_sign: bool = False,
        smooth_sign_scale: float = 5.0,
    ):
        super().__init__()
        self.params = params
        self.num_iterations = int(num_iterations)
        self.offset = float(offset)
        self.smooth_sign = bool(smooth_sign)
        self.smooth_sign_scale = float(smooth_sign_scale)
        self.register_buffer(
            "_edge_vars", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_edge_checks", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_check_edge_ids", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_check_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )

    def _ensure_graph(self, device: torch.device) -> None:
        if self._edge_vars.numel() and self._edge_vars.device == device:
            return
        edge_vars, edge_checks, check_edge_ids, check_mask = torch_edges(
            self.params, device
        )
        self._edge_vars = edge_vars
        self._edge_checks = edge_checks
        self._check_edge_ids = check_edge_ids
        self._check_mask = check_mask

    def forward(
        self, prior_llr: torch.Tensor, num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        p = self.params
        if prior_llr.dim() != 3 or prior_llr.shape[1:] != (p.c, p.n_full):
            raise ValueError(f"prior_llr must have shape [batch, {p.c}, {p.n_full}]")

        batch, c, _ = prior_llr.shape
        flat_prior = prior_llr.reshape(batch * c, p.n_full)
        decoded = self._decode_flat(flat_prior, num_iterations)
        return decoded.view(batch, c, p.n_full)

    def _decode_flat(
        self, prior_llr: torch.Tensor, num_iterations: Optional[int]
    ) -> torch.Tensor:
        self._ensure_graph(prior_llr.device)
        iterations = (
            self.num_iterations if num_iterations is None else int(num_iterations)
        )
        edge_vars = self._edge_vars
        check_edge_ids = self._check_edge_ids
        check_mask = self._check_mask
        batch = prior_llr.shape[0]
        n_edges = edge_vars.numel()

        c2v = torch.zeros(
            batch, n_edges, dtype=prior_llr.dtype, device=prior_llr.device
        )
        llr_vn = prior_llr
        valid_flat_edges = check_edge_ids[check_mask]

        for _ in range(iterations):
            v2c = llr_vn.index_select(1, edge_vars) - c2v
            padded = torch.zeros(
                batch,
                check_edge_ids.shape[0],
                check_edge_ids.shape[1],
                dtype=prior_llr.dtype,
                device=prior_llr.device,
            )
            safe_ids = check_edge_ids.clamp_min(0)
            padded = v2c.index_select(1, safe_ids.reshape(-1)).view_as(padded)
            padded = torch.where(
                check_mask.unsqueeze(0), padded, torch.zeros_like(padded)
            )

            if self.smooth_sign:
                signs = torch.tanh(self.smooth_sign_scale * padded)
                signs = torch.where(
                    check_mask.unsqueeze(0), signs, torch.ones_like(signs)
                )
                sign_prod = signs.prod(dim=2, keepdim=True)
                sign_excl = sign_prod / torch.clamp(signs, min=-1.0, max=1.0).add(
                    (signs.abs() < 1e-7).to(signs.dtype) * 1e-7
                )
            else:
                signs = torch.where(
                    padded >= 0, torch.ones_like(padded), -torch.ones_like(padded)
                )
                signs = torch.where(
                    check_mask.unsqueeze(0), signs, torch.ones_like(signs)
                )
                sign_prod = signs.prod(dim=2, keepdim=True)
                sign_excl = sign_prod * signs

            abs_vals = padded.abs()
            inf = torch.full_like(abs_vals, torch.finfo(abs_vals.dtype).max)
            abs_masked = torch.where(check_mask.unsqueeze(0), abs_vals, inf)
            two_min = torch.topk(abs_masked, k=2, dim=2, largest=False)
            min1 = two_min.values[:, :, 0:1]
            min2 = two_min.values[:, :, 1:2]
            idx1 = two_min.indices[:, :, 0:1]
            pos = torch.arange(check_edge_ids.shape[1], device=prior_llr.device).view(
                1, 1, -1
            )
            min_excl = torch.where(pos == idx1, min2, min1)
            out_padded = sign_excl * F.relu(min_excl - self.offset)
            out_padded = torch.where(
                check_mask.unsqueeze(0), out_padded, torch.zeros_like(out_padded)
            )

            new_c2v = torch.zeros_like(c2v)
            new_c2v.scatter_(
                1,
                valid_flat_edges.unsqueeze(0).expand(batch, -1),
                out_padded[:, check_mask],
            )
            c2v = new_c2v

            llr_vn = prior_llr + torch.zeros_like(prior_llr).scatter_add_(
                1,
                edge_vars.unsqueeze(0).expand(batch, -1),
                c2v,
            )

        return llr_vn
