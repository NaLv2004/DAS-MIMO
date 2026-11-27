"""Measure decoded block error rates on paired, independently generated frames."""

import argparse
import json
from pathlib import Path

import torch
from dasmimo.system import DETECTORS, LEARNED, ROOT, Link, config, detector


@torch.inference_mode()
def run(args):
    if min(args.frames, args.batch) < 1:
        raise ValueError("frames and batch must be positive")
    cfg = config("das_sparse", args.device)
    cfg.Nr, cfg.Nt = args.nr, args.nt
    cfg.CSI_ESTIMATION_NMSE = args.nmse
    link = Link(cfg, args.rate, args.snr)
    models = {}
    for name in args.models:
        spec = config(name, args.device)
        spec.Nr, spec.Nt = args.nr, args.nt
        path = args.weights / (name + ".pt") if name in LEARNED else None
        model = detector(name, spec, path)
        if model is not None:
            model.eval()
        models[name] = (model, spec)
    counts = dict.fromkeys(models, 0)
    for begin in range(0, args.frames, 256):
        torch.manual_seed(args.seed + begin)
        chunk = link.generator.create_compact_latent_dataset_frames(256, args.snr).tensors
        end = min(256, args.frames - begin)
        for offset in range(0, end, args.batch):
            batch = link.materialize(tuple(x[offset:min(offset + args.batch, end)] for x in chunk))
            for name, (model, spec) in models.items():
                info = link.information_llr(
                    name, link.output(name, model, batch, spec), batch[-2]
                )
                counts[name] += int(((info < 0) != batch[5].bool()).any(dim=1).sum())
    result = {
        "rate": args.rate,
        "information_bits": batch[5].shape[-1],
        "ebn0_db": args.snr,
        "seed": args.seed,
        "frames": args.frames,
        "antennas": [args.nr, args.nt],
        "nmse": args.nmse,
        "results": {
            k: {"errors": n, "bler": n / args.frames} for k, n in counts.items()
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", choices=DETECTORS, default=["das_sparse", "das_dense"]
    )
    p.add_argument("--rate", type=float, default=0.8)
    p.add_argument("--snr", type=float, default=10.1)
    p.add_argument("--frames", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--nr", type=int, default=16)
    p.add_argument("--nt", type=int, default=8)
    p.add_argument("--nmse", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=43001)
    p.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path)
    run(p.parse_args())
