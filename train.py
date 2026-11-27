"""Train a detector on the coded MIMO task bank."""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from dasmimo.coding.tasks import build_ldpc_task_specs
from dasmimo.system import DAS, LEARNED, ROOT, Link, config, detector
from torch.nn.parallel import DistributedDataParallel


class Objective(torch.nn.Module):
    def __init__(self, name, net, cfg):
        super().__init__()
        self.name, self.net, self.cfg = name, net, cfg
        self.bank = [s.descriptor for s in build_ldpc_task_specs(cfg)]

    def forward(self, link, compact, epoch):
        mix = min((epoch - 1) / 99, 1.0)
        return link.training_loss(
            self.name,
            self.net,
            link.materialize(compact),
            self.cfg,
            self.bank,
            epoch,
            (1.0 - mix, mix),
        )


def save(value, path):
    tmp = path.with_suffix(".part")
    torch.save(value, tmp)
    tmp.replace(path)


def normalize(args):
    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    device = torch.device(args.device)
    if world > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def run(args):
    rank, world, device = normalize(args)
    name = args.model
    cfg = config(name, device)
    torch.manual_seed(cfg.RANDOM_SEED)
    data_config = name if name in DAS else "das_sparse"
    folder = args.data or Path("data") / data_config
    manifest = json.loads((folder / "manifest.json").read_text())
    specs = build_ldpc_task_specs(cfg)
    if manifest["config"] != data_config or any(
        (row["rate"], row["snr"])
        != (specs[row["task"]].code_rate, specs[row["task"]].ebn0_db)
        for row in manifest["tasks"]
    ):
        raise ValueError("Training data do not match the detector task bank")
    init = args.init
    if name == "detnet" and init is None:
        init = ROOT / "checkpoints" / "detnet_init.pt"
    net = detector(name, cfg, init)
    objective = Objective(name, net, cfg)
    if world > 1:
        objective = DistributedDataParallel(
            objective,
            broadcast_buffers=False,
            find_unused_parameters=True,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    per_rank = args.batch or (
        cfg.BATCH_SIZE // 32 if name in DAS else 700 if name == "admm_net" else 2400
    )
    lr = args.lr or (
        cfg.LR if name in DAS else 0.0005 if name == "channelnet" else 0.0001
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    out = args.output or Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    start = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if (state["model"], state["dataset"]) != (name, manifest):
            raise ValueError("The saved run uses a different model or dataset")
        opt.load_state_dict(state["optimizer"])
        net.load_state_dict(state["weights"])
        start = state["epoch"] + 1
    links = {}
    for epoch in range(start, args.epochs + 1):
        seed = cfg.RANDOM_SEED + 1000003 * epoch
        order = torch.Generator().manual_seed(seed)
        if name in DAS:
            opt.param_groups[0]["lr"] = (
                cfg.LR_MIN
                + (lr - cfg.LR_MIN)
                * (1 + math.cos(math.pi * min(epoch - 1, cfg.EPOCHS) / cfg.EPOCHS))
                / 2
            )
        total = torch.zeros((), device=device)
        frames = 0
        rows = manifest["tasks"]
        for i in torch.randperm(len(rows), generator=order).tolist():
            row = rows[i]
            t = row["task"]
            if t not in links:
                links[t] = Link(cfg, row["rate"], row["snr"])
            tensors = torch.load(
                folder / row["file"], map_location="cpu", weights_only=True, mmap=True
            )
            for batch in torch.randperm(row["frames"], generator=order).split(
                per_rank * world
            ):
                opt.zero_grad(set_to_none=True)
                if name in DAS:
                    net.step_counter.add_(1)
                step_seed = int(torch.randint(0, 2**31 - 1, (), generator=order)) + rank
                width = world * args.microbatch
                for offset in range(0, len(batch), width):
                    local = batch[offset : offset + width][rank::world]
                    n = len(local)
                    if not n:
                        local = batch[:1]
                    compact = tuple(x[local] for x in tensors)
                    sync = world == 1 or offset + width >= len(batch)
                    if name in DAS:
                        torch.manual_seed(step_seed)
                    with nullcontext() if sync else objective.no_sync():
                        loss = objective(links[t], compact, epoch)
                        (loss * (n * world / len(batch))).backward()
                    total += loss.detach() * n
                    if name in DAS:
                        net.detach_code_architecture_state()
                grad = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 10.0, error_if_nonfinite=True
                )
                opt.step()
                frames += len(batch)
        if world > 1:
            dist.all_reduce(total)
        if rank == 0:
            save(net.state_dict(), out / (name + ".pt"))
            save(
                dict(
                    model=name,
                    dataset=manifest,
                    epoch=epoch,
                    weights=net.state_dict(),
                    optimizer=opt.state_dict(),
                ),
                out / "last.pt",
            )
            print(
                json.dumps(
                    dict(
                        epoch=epoch,
                        frames=frames,
                        loss=total.item() / frames,
                        gradient_norm=float(grad),
                        lr=opt.param_groups[0]["lr"],
                    )
                ),
                flush=True,
            )
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=LEARNED)
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch", type=int, help="Codewords per GPU and optimizer step")
    p.add_argument(
        "--microbatch", type=int, default=32, help="Codewords per backward pass per GPU"
    )
    p.add_argument("--lr", type=float)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--init", type=Path, help="Initial detector weights")
    group.add_argument("--resume", type=Path, help="Resume last.pt at the next epoch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if (
        args.microbatch < 1
        or args.epochs < 1
        or (args.batch is not None and args.batch < 1)
    ):
        p.error("Batch sizes and epoch count must be positive")
    run(args)
