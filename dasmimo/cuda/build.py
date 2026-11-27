"""Export and compile the grouped-channel detector kernels."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from ..coding.tasks import build_ldpc_task_specs
from ..model import _ACTIVE_CODE_ARCH_LATENT
from ..system import DAS, ROOT, config, detector
from . import channelnet
from .formula import TypedFormulaExtractor
from .grouped import transform
from .packed import emit_kernel

KERNELS = Path(__file__).parent / "kernels"


def source(name, task, weights, warps=4, aligned=False):
    if name in DAS:
        cfg = config(name, "cpu")
        net = detector(name, cfg, weights).eval()
        spec = build_ldpc_task_specs(cfg)[task]
        with torch.no_grad():
            latent = net.code_arch_encoder(spec.descriptor)[0]
            token = _ACTIVE_CODE_ARCH_LATENT.set(latent)
            try:
                graph = TypedFormulaExtractor(net, 0.01).extract()
            finally:
                _ACTIVE_CODE_ARCH_LATENT.reset(token)
        sparse = name == "das_sparse"
        src = emit_kernel(
            graph,
            task,
            spec.code_rate,
            warps,
            "tanh4" if sparse else "stable",
            sparse,
            aligned,
        )
        return transform(src, task, aligned, warps)
    if name == "channelnet":
        return channelnet.generate(weights)
    if name in ("mmse", "detnet", "admm_net"):
        return (KERNELS / "learned.cu").read_text()
    if name == "kbest8":
        return (KERNELS / "kbest.cu").read_text()
    return (KERNELS / "amp_ep.cu").read_text()


def compile_source(src, path, flags):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc must be on PATH")
    arch = os.getenv("CUDA_ARCH")
    if not arch:
        arch = "".join(map(str, torch.cuda.get_device_capability()))
    options = [
        "-O3",
        "-std=c++17",
        f"-arch=sm_{arch}",
        "--shared",
        "-Xcompiler=-fPIC",
        *flags,
    ]
    signature = hashlib.sha256(
        src.encode()
        + repr(options).encode()
        + subprocess.check_output([nvcc, "--version"])
    ).hexdigest()[:16]
    cu = path.with_suffix(".cu")
    lib = path.with_name(path.name + "-" + signature + ".so")
    cu.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cu.parent) as tmp:
        local_cu = Path(tmp) / "kernel.cu"
        local_cu.write_text(src)
        if not lib.is_file():
            target = Path(tmp) / "kernel.so"
            subprocess.run([nvcc, str(local_cu), *options, "-o", str(target)], check=True)
            target.replace(lib)
        local_cu.replace(cu)
    return lib


def build(name, task, weights, folder, warps=4, compile=True, aligned=False):
    weights = weights or ROOT / "checkpoints" / (name + ".pt")
    stem = f"{name}-task{task}-w{warps}" if name in DAS else name
    if aligned:
        stem += "-aligned"
    src = source(name, task, weights, warps, aligned)
    path = Path(folder) / stem
    flags = (
        ["-lcublasLt"]
        if name in ("mmse", "admm_net", "detnet")
        else ["--use_fast_math"]
    )
    if name.startswith("ep"):
        flags.append(f"-DEP_ITERATIONS={name[2:]}")
    if compile:
        return compile_source(src, path, flags)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".cu")
    path.write_text(src)
    return path
