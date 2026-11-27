from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .params import infer_ldpc_params


@dataclass(frozen=True)
class LDPCCodeTaskSpec:
    task_index: int
    codeword_length: int
    code_rate: float
    ebn0_db: float
    train_frames: int
    eval_frames: int
    descriptor: tuple[float, float, float, float]
    base_graph: int
    lifting_size: int

    def as_dict(self) -> dict:
        return {
            "task_index": self.task_index,
            "codeword_length": self.codeword_length,
            "code_rate": self.code_rate,
            "ebn0_db": self.ebn0_db,
            "train_frames": self.train_frames,
            "eval_frames": self.eval_frames,
            "descriptor": list(self.descriptor),
            "base_graph": self.base_graph,
            "lifting_size": self.lifting_size,
        }


def _allocate_exact(total: int, count: int) -> list[int]:
    if total < count:
        raise ValueError("The total number of codewords must be >= task count")
    base, remainder = divmod(int(total), int(count))
    return [base + (index < remainder) for index in range(count)]


def _interpolate_rate_snr(rate: float, knots: Iterable[Iterable[float]]) -> float:
    pairs = sorted((float(row[0]), float(row[1])) for row in knots)
    if len(pairs) < 2:
        raise ValueError("MULTICODE_SNR_RATE_KNOTS requires at least two points")
    rates = np.asarray([row[0] for row in pairs], dtype=np.float64)
    snrs = np.asarray([row[1] for row in pairs], dtype=np.float64)
    return float(np.interp(float(rate), rates, snrs))


def code_descriptor(
    codeword_length: int,
    code_rate: float,
    base_graph: int,
    lifting_size: int,
    length_min: int,
    length_max: int,
    rate_min: float,
    rate_max: float,
) -> tuple[float, float, float, float]:
    rate_value = 2.0 * (float(code_rate) - rate_min) / (rate_max - rate_min) - 1.0
    log_min, log_max = math.log2(length_min), math.log2(length_max)
    length_value = (
        0.0
        if length_min == length_max
        else 2.0 * (math.log2(codeword_length) - log_min) / (log_max - log_min) - 1.0
    )
    graph_value = 1.0 if int(base_graph) == 1 else -1.0
    # 5G lifting factors span 2..384.  Clipping only guards future parameter
    # extensions; every currently valid task lies inside this interval.
    lifting_value = (
        2.0 * math.log2(max(float(lifting_size), 2.0) / 2.0) / math.log2(384.0 / 2.0)
        - 1.0
    )
    lifting_value = min(max(lifting_value, -1.0), 1.0)
    return (
        float(rate_value),
        float(length_value),
        float(graph_value),
        float(lifting_value),
    )


def build_ldpc_task_specs(cfg) -> list[LDPCCodeTaskSpec]:
    """Create one deterministic stratified sample over the continuous domain."""
    task_count = int(getattr(cfg, "MULTICODE_TASK_COUNT", 24))
    length_min = int(getattr(cfg, "MULTICODE_CODEWORD_LENGTH_MIN", 512))
    length_max = int(getattr(cfg, "MULTICODE_CODEWORD_LENGTH_MAX", 2048))
    rate_min = float(getattr(cfg, "MULTICODE_CODE_RATE_MIN", 0.3))
    rate_max = float(getattr(cfg, "MULTICODE_CODE_RATE_MAX", 0.8))
    train_total = int(getattr(cfg, "MULTICODE_TRAIN_CODEWORDS", 156250))
    eval_total = int(getattr(cfg, "MULTICODE_EVAL_CODEWORDS", max(task_count, 1536)))
    if task_count <= 1:
        raise ValueError("MULTICODE_TASK_COUNT must exceed one")
    if length_min <= 0 or length_max < length_min:
        raise ValueError("Invalid multi-code codeword-length interval")
    if not (0.0 < rate_min < rate_max < 1.0):
        raise ValueError("Invalid multi-code rate interval")
    bits_per_block = int(cfg.Nt) * int(round(math.log2(cfg.ModType)))
    if length_min % bits_per_block or length_max % bits_per_block:
        raise ValueError(
            "Multi-code length endpoints must be divisible by Nt*log2(ModType)"
        )

    seed = int(getattr(cfg, "MULTICODE_TASK_SEED", getattr(cfg, "RANDOM_SEED", 0)))
    rng = np.random.default_rng(seed)
    if length_min == length_max:
        lengths = np.full(task_count, length_min, dtype=np.int64)
    else:
        length_edges = np.linspace(length_min, length_max, task_count + 1)
        raw_lengths = rng.uniform(length_edges[:-1], length_edges[1:])
        lengths = (
            np.rint(raw_lengths / bits_per_block).astype(np.int64) * bits_per_block
        )
        lengths = np.clip(lengths, length_min, length_max)
        # Explicit endpoint coverage prevents a finite stratified bank from missing
        # the limits of the claimed generalization interval.
        lengths[0], lengths[-1] = length_min, length_max
        rng.shuffle(lengths)

    rate_edges = np.linspace(rate_min, rate_max, task_count + 1)
    rates = rng.uniform(rate_edges[:-1], rate_edges[1:])
    rates[0], rates[-1] = rate_min, rate_max
    rng.shuffle(rates)

    train_allocations = _allocate_exact(train_total, task_count)
    eval_allocations = _allocate_exact(eval_total, task_count)
    knots = getattr(
        cfg,
        "MULTICODE_SNR_RATE_KNOTS",
        ((0.30, 7.5), (0.45, 8.3), (0.60, 9.0), (0.80, 10.5)),
    )
    length_slope = float(getattr(cfg, "MULTICODE_SNR_LENGTH_DB_PER_OCTAVE", -0.4))
    length_reference = float(getattr(cfg, "MULTICODE_SNR_REFERENCE_LENGTH", 1024))

    specs = []
    for task_index, (length, rate) in enumerate(zip(lengths.tolist(), rates.tolist())):
        length = int(length)
        rate = float(rate)
        params = infer_ldpc_params(
            length,
            rate,
            rv=int(getattr(cfg, "DECODER_AWARE_RV", 0)),
            qm=int(getattr(cfg, "DECODER_AWARE_LDPC_QM", 1)),
            nlayers=int(getattr(cfg, "DECODER_AWARE_LAYERS", 1)),
        )
        snr = _interpolate_rate_snr(rate, knots)
        snr += length_slope * math.log2(length / length_reference)
        descriptor = code_descriptor(
            length,
            rate,
            params.bgn,
            params.zc,
            length_min,
            length_max,
            rate_min,
            rate_max,
        )
        specs.append(
            LDPCCodeTaskSpec(
                task_index=task_index,
                codeword_length=length,
                code_rate=rate,
                ebn0_db=float(snr),
                train_frames=train_allocations[task_index],
                eval_frames=eval_allocations[task_index],
                descriptor=descriptor,
                base_graph=params.bgn,
                lifting_size=params.zc,
            )
        )
    return specs
