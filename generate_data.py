"""Generate fixed 5G LDPC training frames for the detector task bank."""

import argparse
import json
from pathlib import Path

import torch
from dasmimo.coding.tasks import build_ldpc_task_specs
from dasmimo.system import DAS, Link, config


@torch.inference_mode()
def generate(args):
    cfg = config(args.config, args.device)
    specs = build_ldpc_task_specs(cfg)
    tasks = (
        sorted(set(args.tasks)) if args.tasks is not None else list(range(len(specs)))
    )
    seed = cfg.RANDOM_SEED if args.seed is None else args.seed
    rows = []
    for task in tasks:
        spec = specs[task]
        frames = (
            spec.train_frames if args.frames_per_task is None else args.frames_per_task
        )
        if frames < 1:
            raise ValueError("Each task needs at least one frame")
        rows.append(
            dict(
                task=task,
                rate=spec.code_rate,
                snr=spec.ebn0_db,
                frames=frames,
                file=f"task-{task:02d}.pt",
            )
        )
    manifest = dict(
        config=args.config,
        seed=seed,
        antennas=[cfg.Nr, cfg.Nt],
        codeword_length=1024,
        tasks=rows,
    )
    folder = args.output or Path("data") / args.config
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "manifest.json"
    if metadata.exists() and json.loads(metadata.read_text()) != manifest:
        raise ValueError(f"{folder} contains a different dataset")
    for row in rows:
        dest = folder / row["file"]
        if metadata.exists() and dest.is_file():
            continue
        torch.manual_seed(seed + row["task"])
        link = Link(cfg, row["rate"], row["snr"])
        dataset = link.generator.create_compact_latent_dataset_frames(
            row["frames"], row["snr"]
        )
        tmp = dest.with_suffix(".part")
        torch.save(dataset.tensors, tmp)
        tmp.replace(dest)
        print(
            f"Task {row['task']:2d}  R={row['rate']:.4f}  {row['frames']} frames",
            flush=True,
        )
    tmp = metadata.with_suffix(".part")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(metadata)
    print(metadata)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=DAS, default="das_sparse")
    p.add_argument("--tasks", type=int, nargs="+", choices=range(24))
    p.add_argument("--frames-per-task", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate(p.parse_args())
