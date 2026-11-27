import math
from typing import Optional

import torch

CPP_TABLES = {
    2: (-0.707107, 0.707107),
    4: (-0.316228, -0.948683, 0.316228, 0.948683),
    6: (
        -0.462910,
        -0.154303,
        -0.771517,
        -1.08012,
        0.462910,
        0.154303,
        0.771517,
        1.08012,
    ),
    8: (
        -0.383482,
        -0.536875,
        -0.230089,
        -0.07669,
        -0.843661,
        -0.690268,
        -0.997054,
        -1.15044,
        0.383482,
        0.536875,
        0.230089,
        0.07669,
        0.843661,
        0.690268,
        0.997054,
        1.15044,
    ),
}


def cpp_modulate_bits(bits: torch.Tensor, mode_type: int = 4) -> torch.Tensor:
    """Reference-code modulation for mode_type bits per complex symbol.

    Returns complex symbols represented as a real tensor [batch, n_symbols, 2],
    where the last dimension is [real, imag].
    """
    if mode_type not in CPP_TABLES:
        raise ValueError("mode_type must be 2, 4, 6, or 8")
    if bits.dim() != 2 or bits.shape[1] % mode_type != 0:
        raise ValueError("bits must have shape [batch, n_symbols * mode_type]")

    half = mode_type // 2
    batch = bits.shape[0]
    n_symbols = bits.shape[1] // mode_type
    chunks = bits.to(torch.long).view(batch, n_symbols, mode_type)
    powers = (
        2 ** torch.arange(half - 1, -1, -1, device=bits.device, dtype=torch.long)
    ).view(1, 1, half)
    idx_i = (chunks[:, :, :half] * powers).sum(dim=2)
    idx_q = (chunks[:, :, half:] * powers).sum(dim=2)
    table = torch.as_tensor(
        CPP_TABLES[mode_type], dtype=torch.float32, device=bits.device
    )
    return torch.stack([table[idx_i], table[idx_q]], dim=2)


def cpp_modulate_bits_differentiable(
    bits: torch.Tensor, mode_type: int = 4
) -> torch.Tensor:
    """Differentiable multilinear extension of :func:`cpp_modulate_bits`.

    At binary inputs this returns exactly the same table entries as the C++
    reference mapper.  For straight-through bit-routing surrogates it accepts
    continuous values and preserves gradients to every label bit.
    """
    if mode_type not in CPP_TABLES:
        raise ValueError("mode_type must be 2, 4, 6, or 8")
    if bits.dim() != 2 or bits.shape[1] % mode_type != 0:
        raise ValueError("bits must have shape [batch, n_symbols * mode_type]")

    half = mode_type // 2
    chunks = bits.view(bits.shape[0], -1, mode_type)
    levels = 1 << half
    binary_labels = (
        torch.arange(levels, device=bits.device, dtype=torch.long).view(levels, 1)
        >> torch.arange(half - 1, -1, -1, device=bits.device, dtype=torch.long).view(
            1, half
        )
    ) & 1
    binary_labels = binary_labels.to(bits.dtype)
    table = torch.as_tensor(CPP_TABLES[mode_type], dtype=bits.dtype, device=bits.device)

    def _axis_value(axis_bits):
        labels = binary_labels.view(1, 1, levels, half)
        values = axis_bits.unsqueeze(2)
        interpolation = torch.where(labels > 0.5, values, 1.0 - values).prod(dim=-1)
        return (interpolation * table.view(1, 1, levels)).sum(dim=-1)

    real = _axis_value(chunks[:, :, :half])
    imag = _axis_value(chunks[:, :, half:])
    return torch.stack([real, imag], dim=2)


def cpp_soft_demod_complex(symbols: torch.Tensor, mode_type: int = 4) -> torch.Tensor:
    if symbols.dim() != 3 or symbols.shape[2] != 2:
        raise ValueError("symbols must have shape [batch, n_symbols, 2]")
    real = symbols[:, :, 0]
    imag = symbols[:, :, 1]
    if mode_type == 2:
        return torch.stack([real, imag], dim=2).reshape(symbols.shape[0], -1)
    if mode_type == 4:
        return torch.stack(
            [real, real.abs() - 0.6324555, imag, imag.abs() - 0.6324555], dim=2
        ).reshape(symbols.shape[0], -1)
    if mode_type == 6:
        l0 = real
        l1 = l0.abs() - 0.6172134
        l2 = l1.abs() - 0.3086067
        l3 = imag
        l4 = l3.abs() - 0.6172134
        l5 = l4.abs() - 0.3086067
        return torch.stack([l0, l1, l2, l3, l4, l5], dim=2).reshape(
            symbols.shape[0], -1
        )
    if mode_type == 8:
        l0 = real
        l1 = l0.abs() - 0.6135715
        l2 = l1.abs() - 0.306786
        l3 = l2.abs() - 0.1533895
        l4 = imag
        l5 = l4.abs() - 0.6135715
        l6 = l5.abs() - 0.306786
        l7 = l6.abs() - 0.1533895
        return torch.stack([l0, l1, l2, l3, l4, l5, l6, l7], dim=2).reshape(
            symbols.shape[0], -1
        )
    raise ValueError("mode_type must be 2, 4, 6, or 8")


def cpp_soft_demod_real_mimo(
    x_est: torch.Tensor, nt: int, mode_type: int = 4
) -> torch.Tensor:
    """C++ soft_demodulation for detector output ordered [real antennas, imag antennas].

    The returned LLR is positive when the corresponding modulation bit is more
    likely to be 1.
    """
    if x_est.dim() == 3 and x_est.shape[2] == 1:
        x = x_est.squeeze(2)
    elif x_est.dim() == 2:
        x = x_est
    else:
        raise ValueError("x_est must have shape [batch, 2*Nt] or [batch, 2*Nt, 1]")
    if x.shape[1] != 2 * nt:
        raise ValueError(f"x_est second dimension must be {2 * nt}")

    real = x[:, :nt]
    imag = x[:, nt:]
    symbols = torch.stack([real, imag], dim=2)
    return cpp_soft_demod_complex(symbols, mode_type=mode_type)


def cpp_symbol_llr_to_decoder_llr(
    symbol_llr: torch.Tensor, scale: float = 32.0
) -> torch.Tensor:
    """Convert C++ demod LLR convention to LDPC decoder prior convention.

    The demapper multiplies its bit score by the LLR scale and negates it before
    LDPC rate recovery. Positive decoder LLR therefore means bit 0.
    """
    return -float(scale) * symbol_llr


def apply_cpp_interleaver(
    bits_or_llr: torch.Tensor, rows: Optional[int]
) -> torch.Tensor:
    if rows is None or rows <= 0:
        return bits_or_llr
    if bits_or_llr.shape[1] % rows != 0:
        raise ValueError(
            f"Length {bits_or_llr.shape[1]} is not divisible by interleaver rows={rows}"
        )
    cols = bits_or_llr.shape[1] // rows
    return (
        bits_or_llr.view(bits_or_llr.shape[0], cols, rows)
        .transpose(1, 2)
        .reshape(bits_or_llr.shape[0], -1)
    )


def apply_cpp_deinterleaver(
    bits_or_llr: torch.Tensor, rows: Optional[int]
) -> torch.Tensor:
    if rows is None or rows <= 0:
        return bits_or_llr
    if bits_or_llr.shape[1] % rows != 0:
        raise ValueError(
            f"Length {bits_or_llr.shape[1]} is not divisible by interleaver rows={rows}"
        )
    cols = bits_or_llr.shape[1] // rows
    return (
        bits_or_llr.view(bits_or_llr.shape[0], rows, cols)
        .transpose(1, 2)
        .reshape(bits_or_llr.shape[0], -1)
    )


def fixed_random_interleaver_indices(length: int, seed: int, device=None):
    """Build a reproducible coded-bit permutation and its exact inverse."""
    if int(length) <= 0:
        raise ValueError("interleaver length must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(length), generator=generator, dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(int(length), dtype=torch.long)
    target = torch.device(device or "cpu")
    return permutation.to(target), inverse.to(target)


def apply_fixed_random_interleaver(
    bits_or_llr: torch.Tensor, permutation: Optional[torch.Tensor]
):
    if permutation is None:
        return bits_or_llr
    if bits_or_llr.dim() != 2 or bits_or_llr.shape[1] != permutation.numel():
        raise ValueError(
            f"random interleaver expects [batch, {permutation.numel()}], "
            f"got {tuple(bits_or_llr.shape)}"
        )
    return bits_or_llr.index_select(1, permutation.to(bits_or_llr.device))


def apply_fixed_random_deinterleaver(
    bits_or_llr: torch.Tensor, inverse: Optional[torch.Tensor]
):
    if inverse is None:
        return bits_or_llr
    if bits_or_llr.dim() != 2 or bits_or_llr.shape[1] != inverse.numel():
        raise ValueError(
            f"random deinterleaver expects [batch, {inverse.numel()}], "
            f"got {tuple(bits_or_llr.shape)}"
        )
    return bits_or_llr.index_select(1, inverse.to(bits_or_llr.device))


def mode_type_from_mod_type(mod_type: int) -> int:
    value = int(round(math.log2(mod_type)))
    if 2**value != mod_type:
        raise ValueError("mod_type must be a power of two")
    return value
