import argparse
from pathlib import Path

from dasmimo.cuda.build import build
from dasmimo.system import DETECTORS

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export a detector's CUDA source and optionally compile it."
    )
    p.add_argument("model", choices=DETECTORS)
    p.add_argument("--task", type=int, choices=range(24), default=10)
    p.add_argument("--warps", type=int, choices=(2, 4), default=4)
    p.add_argument("--weights", type=Path)
    p.add_argument("--build-dir", type=Path, default=Path("build"))
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--aligned",
        action="store_true",
        help="Specialize for reuse counts divisible by eight times the warp count",
    )
    args = p.parse_args()
    print(
        build(
            args.model,
            args.task,
            args.weights,
            args.build_dir,
            args.warps,
            args.compile,
            aligned=args.aligned,
        )
    )
