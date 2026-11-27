"""Measure channel preprocessing and symbol detection with CUDA events."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
from statistics import median

from dasmimo.cuda.verify import compare

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import torch
from dasmimo.cuda.build import KERNELS, build, compile_source
from dasmimo.cuda.learned import Networks
from dasmimo.cuda.runtime import GroupedTimer, configure_grouped, make_inputs
from dasmimo.system import DAS, DETECTORS, ROOT, config, detector


def candidates(name, library, channels, reuse, warps, aligned):
    if name in DAS:
        symbols = [("launch_grouped_dag_full", 8 * warps)]
    elif name.startswith("amp"):
        symbols = [
            (f"launch_grouped_amp_t{name[3:]}_{t}_full", t // 8) for t in (64, 128)
        ]
    elif name.startswith("ep"):
        symbols = [(f"launch_grouped_ep_{t}_full", t // 16) for t in (64, 96, 128)]
    elif name == "kbest8":
        symbols = [("launch_grouped_kbest_8_full", 16)]
    elif name == "channelnet":
        symbols = [(f"launch_grouped_channelnet_w{w}_full", w) for w in range(1, 9)]
    else:
        symbols = [("mmse_full", 8)]
    for sym, vectors in symbols:
        fn = getattr(library, sym)
        configure_grouped(fn)
        limit = math.ceil(reuse / vectors)
        for grid in sorted({min(limit, g) for g in (1, 2, 4, 8, 16, 32, 64, limit)}):
            yield (
                (
                    f"w{warps}_{sym}" + ("_aligned" if aligned else "")
                    if name in DAS
                    else sym
                ),
                fn,
                grid,
            )


def buffers(name, p, m, seed):
    inputs = make_inputs(p, m, seed, torch.device("cuda"))
    if name.startswith("amp"):
        sizes = [256, 128, 1]
    elif name.startswith("ep"):
        sizes = [256, 256, 128, 8]
    elif name == "kbest8":
        sizes = [136, 136 * 4]
    elif name == "channelnet":
        sizes = []
    elif name == "mmse":
        sizes = [1024]
    else:
        sizes = [256, 128, 8, 128, 2]
    return (
        *inputs[:4],
        *(
            torch.empty(max(1, p * x), device="cuda")
            for x in sizes + [0] * (6 - len(sizes))
        ),
    )


@torch.inference_mode()
def measure(args):
    timer = GroupedTimer(
        compile_source((KERNELS / "timer.cu").read_text(), args.build_dir / "timer", [])
    )
    results = []
    for name in args.models:
        tasks = args.tasks if name in DAS else [-1]
        weights = args.weights / (name + ".pt")
        for task in tasks:
            libraries = [
                (
                    w,
                    aligned,
                    ctypes.CDLL(
                        str(
                            build(
                                name, task, weights, args.build_dir, w, aligned=aligned
                            ).resolve()
                        )
                    ),
                )
                for w, aligned in (
                    [(4, True), (4, False)]
                    if name == "das_sparse"
                    else [(2, False), (4, False)]
                    if name == "das_dense"
                    else [(4, False)]
                )
            ]
            trained = None
            if name in ("detnet", "admm_net"):
                trained = Networks(
                    libraries[0][2],
                    {name: detector(name, config(name, "cuda"), weights).eval()},
                )

            def setup(p, m):
                seed = 20260831 + 1009 * p + m
                if trained:
                    return trained.inputs(p, m, seed), trained.variants(name, p, m)
                return buffers(name, p, m, seed), [
                    v
                    for w, aligned, lib in libraries
                    if not aligned or m % (8 * w) == 0
                    for v in candidates(name, lib, p, m, w, aligned)
                ]

            check = None
            if args.check:
                buf, variants = setup(2, 64)
                _, fn, g = variants[0]
                check = compare(name, task, weights, fn, buf, 2, 64, g)
                print(
                    json.dumps(dict(model=name, task=task, numerical_check=check)),
                    flush=True,
                )
            for m in args.reuse:
                p = args.channels
                buf, variants = setup(p, m)
                probes = [
                    timer.measure(fn, buf, p, m, g, 5, 20, task)
                    for _, fn, g in variants
                ]
                ranked = sorted(range(len(probes)), key=probes.__getitem__)[:2]
                medians = {}
                for i in ranked:
                    _, fn, g = variants[i]
                    medians[i] = median(
                        timer.measure(fn, buf, p, m, g, 5, 10, task) for _ in range(5)
                    )
                i = min(medians, key=medians.get)
                ms = medians[i]
                row = dict(
                    model=name,
                    task=task if name in DAS else None,
                    channels=p,
                    reuse=m,
                    median_ms=ms,
                    mvectors_s=p * m / (1000 * ms),
                    kernel=variants[i][0],
                    grid=variants[i][2],
                )
                print(json.dumps(row), flush=True)
                if check:
                    row["numerical_check"] = check
                results.append(row)
    aggregate = [
        dict(
            model=name,
            reuse=m,
            median_mvectors_s=median(
                row["mvectors_s"]
                for row in results
                if row["model"] == name and row["reuse"] == m
            ),
        )
        for name in args.models
        for m in args.reuse
    ]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "results": results,
        "median_across_tasks": aggregate,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=DETECTORS, default=list(DAS))
    p.add_argument("--tasks", nargs="+", type=int, choices=range(24), default=[10])
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--reuse", nargs="+", type=int, default=[64, 128, 512])
    p.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--build-dir", type=Path, default=Path("build"))
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--check",
        action="store_true",
        help="Compare CUDA symbols with PyTorch before timing",
    )
    p.add_argument(
        "--all-tasks", action="store_true", help="Benchmark all 24 code configurations"
    )
    args = p.parse_args()
    if args.all_tasks:
        args.tasks = list(range(24))
    measure(args)
