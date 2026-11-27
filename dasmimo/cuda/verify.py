"""Compare the CUDA symbol output against the PyTorch detector."""

import torch

from ..baselines import classical
from ..coding.tasks import build_ldpc_task_specs
from ..system import DAS, LEARNED, config, detector
from .runtime import launch_once, real_equivalent_h


@torch.inference_mode()
def compare(name, task, weights, fn, buf, p, m, grid):
    launch_once(fn, buf, p, m, grid, task)
    h, y, noise, actual = buf[:4]
    H = real_equivalent_h(h).repeat_interleave(m, dim=0)
    y = y.reshape(p * m, 32, 1)
    noise = noise.repeat_interleave(m)
    levels = classical.pam_constellation(16, device=y.device)
    if name in LEARNED:
        cfg = config(name, y.device)
        model = detector(name, cfg, weights).eval()
        if name in DAS:
            gram = H.mT.bmm(H) + noise[:, None, None] * torch.eye(16, device=y.device)
            scale = gram.abs().sum(-1).amax(-1).view(-1, 1, 1)
            code = build_ldpc_task_specs(cfg)[task].descriptor
            reference = model(H, y, gram / scale, scale, True, 0.01, code)[-1]
        else:
            reference = model(H, y).final
    elif name == "mmse":
        reference = classical.mmse_detect(H, y, noise)
    elif name.startswith("amp"):
        reference = classical.io_lama_detect(
            H, y, noise, levels, 16, iterations=int(name[3:]), damping=0.5
        )
    elif name.startswith("ep"):
        reference = classical.ep_detect(
            H, y, noise, levels, iterations=int(name[2:]), delta=0.5
        )
    else:
        reference = classical.kbest_detect(H, y, levels, k=8)[0]
    expected = reference.reshape_as(actual)
    delta = (actual - expected).abs()
    if not delta.isfinite().all():
        raise ValueError(f"{name} produced nonfinite outputs")
    hard = lambda x: (x[..., None] - levels).abs().argmin(-1)
    if not torch.allclose(actual, expected, atol=3e-4, rtol=3e-4):
        raise ValueError(
            f"{name}: CUDA and PyTorch differ by up to {delta.max().item():.3g}"
        )
    return dict(
        coordinates=actual.numel(),
        max_abs=float(delta.max()),
        mean_abs=float(delta.mean()),
        hard_mismatches=int((hard(actual) != hard(expected)).sum()),
    )
