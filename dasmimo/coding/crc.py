import torch


def append_tb_crc(bits: torch.Tensor, crc_len: int) -> torch.Tensor:
    if crc_len == 16:
        crc = _crc_lsb_first(
            bits,
            width=16,
            poly=0x1021,
            init=0x0000,
            xorout=0x0000,
            refin=True,
            refout=True,
        )
    elif crc_len == 24:
        crc = _crc_lsb_first(
            bits,
            width=24,
            poly=0x864CFB,
            init=0x000000,
            xorout=0x000000,
            refin=True,
            refout=True,
        )
    else:
        raise ValueError("crc_len must be 16 or 24")
    return torch.cat([bits, crc.to(bits.device)], dim=1)


def append_cb_crc(bits: torch.Tensor, crc_len: int) -> torch.Tensor:
    if crc_len == 16:
        crc = _crc_lsb_first(
            bits,
            width=16,
            poly=0x1021,
            init=0x0000,
            xorout=0x0000,
            refin=True,
            refout=True,
        )
    elif crc_len == 24:
        crc = _crc_lsb_first(
            bits,
            width=24,
            poly=0x800063,
            init=0x000000,
            xorout=0x000000,
            refin=True,
            refout=True,
        )
    else:
        raise ValueError("crc_len must be 16 or 24")
    return torch.cat([bits, crc.to(bits.device)], dim=1)


def _crc_lsb_first(
    bits: torch.Tensor,
    width: int,
    poly: int,
    init: int,
    xorout: int,
    refin: bool,
    refout: bool,
) -> torch.Tensor:
    """CRC helper matching the reference code bit order: bytes are LSB-first.

    This function is used for target/codeword generation only, so it intentionally
    works on CPU integers rather than participating in the differentiable path.
    """
    if bits.dim() != 2:
        raise ValueError("bits must have shape [batch, n_bits]")

    cpu_bits = bits.detach().to("cpu", dtype=torch.int64)
    out = torch.empty(cpu_bits.shape[0], width, dtype=torch.long)
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    reflected_poly = _reflect(poly, width)

    for row_idx, row in enumerate(cpu_bits):
        reg = init & mask
        for bit in row.tolist():
            b = int(bit) & 1
            if refin:
                feedback = (reg ^ b) & 1
                reg >>= 1
                if feedback:
                    reg ^= reflected_poly
            else:
                feedback = ((reg & top) != 0) ^ bool(b)
                reg = (reg << 1) & mask
                if feedback:
                    reg ^= poly
        if refout ^ refin:
            reg = _reflect(reg, width)
        reg ^= xorout
        reg &= mask
        for i in range(width):
            out[row_idx, i] = (reg >> i) & 1

    return out


def _reflect(value: int, width: int) -> int:
    reflected = 0
    for i in range(width):
        if value & (1 << i):
            reflected |= 1 << (width - 1 - i)
    return reflected
