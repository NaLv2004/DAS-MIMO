import ctypes
import math
from pathlib import Path

import torch


def pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def configure_grouped(function) -> None:
    function.argtypes = (
        [ctypes.c_int] + [ctypes.c_void_p] * 10 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    function.restype = ctypes.c_int


class GroupedTimer:
    def __init__(self, library: Path):
        handle = ctypes.CDLL(str(library.resolve()))
        function = handle.benchmark_grouped_h_launch_loop
        function.argtypes = (
            [ctypes.c_void_p, ctypes.c_int]
            + [ctypes.c_void_p] * 10
            + [ctypes.c_int] * 5
            + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        )
        function.restype = ctypes.c_int
        self.handle = handle
        self.function = function

    @staticmethod
    def stream() -> ctypes.c_void_p:
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def measure(
        self,
        function,
        tensors: tuple[torch.Tensor, ...],
        channels: int,
        vectors: int,
        blocks: int,
        warmups: int,
        replays: int,
        task: int = -1,
    ) -> float:
        elapsed = ctypes.c_float()
        status = self.function(
            ctypes.cast(function, ctypes.c_void_p),
            task,
            *(pointer(tensor) for tensor in tensors),
            int(channels),
            int(vectors),
            int(blocks),
            int(warmups),
            int(replays),
            self.stream(),
            ctypes.byref(elapsed),
        )
        if status:
            raise RuntimeError(f"grouped timing failed with CUDA status {status}")
        return float(elapsed.value)


def make_inputs(
    channels: int,
    vectors: int,
    seed: int,
    device: torch.device,
    nr: int = 16,
    nt: int = 8,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real = torch.randn(
        channels, nr, nt, device=device, generator=generator
    ) / math.sqrt(2.0)
    imag = torch.randn(
        channels, nr, nt, device=device, generator=generator
    ) / math.sqrt(2.0)
    h = torch.stack((real, imag), dim=-1).contiguous()
    y = torch.randn(
        channels, vectors, 2 * nr, device=device, generator=generator
    ).contiguous()
    noise = torch.full((channels,), 0.2, device=device)
    output = torch.empty(channels * vectors, 2 * nt, device=device)
    dummy = tuple(torch.zeros(1, device=device) for _ in range(6))
    return (h, y, noise, output, *dummy)


def real_equivalent_h(h: torch.Tensor) -> torch.Tensor:
    real, imag = h[..., 0], h[..., 1]
    return torch.cat(
        (torch.cat((real, -imag), dim=2), torch.cat((imag, real), dim=2)),
        dim=1,
    ).contiguous()


def launch_once(
    function,
    tensors: tuple[torch.Tensor, ...],
    channels: int,
    vectors: int,
    blocks: int,
    task: int = -1,
) -> None:
    status = function(
        task,
        *(pointer(tensor) for tensor in tensors),
        int(channels),
        int(vectors),
        int(blocks),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status:
        raise RuntimeError(f"CUDA launch returned CUDA status {status}")
