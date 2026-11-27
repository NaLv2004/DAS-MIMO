from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

Z_LIST: Tuple[int, ...] = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    288,
    320,
    352,
    384,
)

Z_SETS: Tuple[Tuple[int, ...], ...] = (
    (2, 4, 8, 16, 32, 64, 128, 256),
    (3, 6, 12, 24, 48, 96, 192, 384),
    (5, 10, 20, 40, 80, 160, 320, 0),
    (7, 14, 28, 56, 112, 224, 0, 0),
    (9, 18, 36, 72, 144, 288, 0, 0),
    (11, 22, 44, 88, 176, 352, 0, 0),
    (13, 26, 52, 104, 208, 0, 0, 0),
    (15, 30, 60, 120, 240, 0, 0, 0),
)


@dataclass(frozen=True)
class LDPC5GParams:
    info_len: int
    code_rate: float
    out_len: int
    tb_crc_len: int
    b_with_crc: int
    bgn: int
    kcb: int
    c: int
    b_cb_bit: int
    k_cb_bit: int
    kb: int
    zc: int
    set_idx: int
    k: int
    num_filler: int
    nb: int
    mb: int
    n_full: int
    m: int
    n_punctured: int
    rv: int
    qm: int
    nlayers: int
    e_array: Tuple[int, ...]
    k0: Tuple[int, ...]

    @property
    def filler_start_full(self) -> int:
        return self.k_cb_bit

    @property
    def filler_end_full(self) -> int:
        return self.k

    @property
    def filler_start_punctured(self) -> int:
        return max(0, self.filler_start_full - 2 * self.zc)

    @property
    def filler_end_punctured(self) -> int:
        return max(0, self.filler_end_full - 2 * self.zc)


def infer_ldpc_params(
    codeword_length: int,
    code_rate: float,
    rv: int = 0,
    qm: int = 1,
    nlayers: int = 1,
) -> LDPC5GParams:
    """Infer the LDPC geometry for the rate-matched length and requested code rate."""
    out_len = int(codeword_length)
    if out_len <= 0:
        raise ValueError("codeword_length must be positive")
    if not (0.0 < float(code_rate) < 1.0):
        raise ValueError("code_rate must be in (0, 1)")
    if rv not in (0, 1, 2, 3):
        raise ValueError("rv must be 0, 1, 2, or 3")

    info_len = int(round(out_len * float(code_rate)))
    tb_crc_len = 24 if info_len > 3824 else 16
    b_with_crc = info_len + tb_crc_len

    if info_len < 293 or (info_len < 3825 and code_rate <= 0.67) or code_rate <= 0.25:
        bgn = 2
        kcb = 3840
    else:
        bgn = 1
        kcb = 8448

    if b_with_crc < kcb:
        c = 1
        b_cb_bit = b_with_crc
    else:
        c = int(np.ceil(b_with_crc / float(kcb - tb_crc_len)))
        b_cb_bit = b_with_crc + c * tb_crc_len
    k_cb_bit = int(np.ceil(b_cb_bit / float(c)))

    if bgn == 1:
        kb = 22
    elif b_with_crc > 640:
        kb = 10
    elif b_with_crc > 560:
        kb = 9
    elif b_with_crc > 192:
        kb = 8
    else:
        kb = 6

    zc = None
    for candidate in Z_LIST:
        if kb * candidate >= k_cb_bit:
            zc = candidate
            break
    if zc is None:
        raise ValueError("No valid lifting factor found for this codeword length/rate")

    set_idx = None
    for row_idx, row in enumerate(Z_SETS, start=1):
        if zc in row:
            set_idx = row_idx
            break
    if set_idx is None:
        raise ValueError(f"No BG set index found for Zc={zc}")

    if bgn == 1:
        k = 22 * zc
        nb = 68
        mb = 46
    else:
        k = 10 * zc
        nb = 52
        mb = 42

    n_full = nb * zc
    m = mb * zc
    n_punctured = n_full - 2 * zc
    num_filler = k - k_cb_bit
    e_array = _infer_e_array(c, out_len, qm, nlayers)
    k0 = tuple(_infer_k0(n_punctured, zc, bgn, rv) for _ in range(c))

    return LDPC5GParams(
        info_len=info_len,
        code_rate=float(code_rate),
        out_len=out_len,
        tb_crc_len=tb_crc_len,
        b_with_crc=b_with_crc,
        bgn=bgn,
        kcb=kcb,
        c=c,
        b_cb_bit=b_cb_bit,
        k_cb_bit=k_cb_bit,
        kb=kb,
        zc=zc,
        set_idx=set_idx,
        k=k,
        num_filler=num_filler,
        nb=nb,
        mb=mb,
        n_full=n_full,
        m=m,
        n_punctured=n_punctured,
        rv=rv,
        qm=qm,
        nlayers=nlayers,
        e_array=tuple(e_array),
        k0=k0,
    )


def _infer_e_array(c: int, out_len: int, qm: int, nlayers: int) -> List[int]:
    e_array = []
    denom = nlayers * qm * c
    threshold = c - (out_len // (nlayers * qm)) % c - 1
    for r in range(c):
        if r <= threshold:
            e = nlayers * qm * int(np.floor(out_len / float(denom)))
        else:
            e = nlayers * qm * int(np.ceil(out_len / float(denom)))
        e_array.append(e)
    if sum(e_array) != out_len:
        raise RuntimeError("Internal E-array inference error")
    return e_array


def _infer_k0(ncb: int, zc: int, bgn: int, rv: int) -> int:
    if rv == 0:
        return 0
    if bgn == 1:
        fractions = {1: 17, 2: 33, 3: 56}
        denom = 66 * zc
    else:
        fractions = {1: 13, 2: 25, 3: 43}
        denom = 50 * zc
    return int(np.floor(fractions[rv] * ncb / float(denom))) * zc


@lru_cache(maxsize=None)
def load_base_graph(bgn: int, set_idx: int) -> np.ndarray:
    if bgn not in (1, 2):
        raise ValueError("bgn must be 1 or 2")
    if not (1 <= set_idx <= 8):
        raise ValueError("set_idx must be in [1, 8]")

    filename = f"BG{bgn}S{set_idx}.txt"
    return np.loadtxt(Path(__file__).parent / "base_graphs" / filename, dtype=np.int64)


@lru_cache(maxsize=None)
def build_ldpc_edges(
    bgn: int, set_idx: int, zc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bg = load_base_graph(bgn, set_idx)
    edge_vars = []
    edge_checks = []
    check_edges: List[List[int]] = [[] for _ in range(bg.shape[0] * zc)]

    for br in range(bg.shape[0]):
        for bc in range(bg.shape[1]):
            shift = int(bg[br, bc])
            if shift < 0:
                continue
            shift %= zc
            for z in range(zc):
                check = br * zc + z
                var = bc * zc + ((z + shift) % zc)
                edge_id = len(edge_vars)
                edge_checks.append(check)
                edge_vars.append(var)
                check_edges[check].append(edge_id)

    max_degree = max(len(row) for row in check_edges)
    padded = np.full((len(check_edges), max_degree), -1, dtype=np.int64)
    mask = np.zeros((len(check_edges), max_degree), dtype=np.bool_)
    for i, row in enumerate(check_edges):
        padded[i, : len(row)] = row
        mask[i, : len(row)] = True

    return (
        np.asarray(edge_vars, dtype=np.int64),
        np.asarray(edge_checks, dtype=np.int64),
        padded,
        mask,
    )


def torch_edges(
    params: LDPC5GParams, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_vars, edge_checks, check_edge_ids, check_mask = build_ldpc_edges(
        params.bgn,
        params.set_idx,
        params.zc,
    )
    return (
        torch.as_tensor(edge_vars, dtype=torch.long, device=device),
        torch.as_tensor(edge_checks, dtype=torch.long, device=device),
        torch.as_tensor(check_edge_ids, dtype=torch.long, device=device),
        torch.as_tensor(check_mask, dtype=torch.bool, device=device),
    )
