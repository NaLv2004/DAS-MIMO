"""Count equivalent FLOPs per detected MIMO vector."""

import argparse
import json
from pathlib import Path

from dasmimo.cuda.build import source
from dasmimo.cuda.cost import baseline_costs, dag_cost
from dasmimo.system import DAS, DETECTORS, ROOT

KEYS = {
    "amp8": "damped_io_lama_t8",
    "amp15": "damped_io_lama_t15",
    "ep7": "ep_t7",
    "ep3": "ep_t3",
    "kbest8": "kbest_k8_maxlog",
    "admm_net": "admm_t10",
    "detnet": "detnet_l10",
    "channelnet": "channelnet_l10",
    "mmse": "mmse",
}


def count(name, task, reuse, weights):
    cost = (
        dag_cost(source(name, task, weights), task)
        if name in DAS
        else baseline_costs()[KEYS[name]]
    )
    pre = cost["pre_arithmetic"] + cost["pre_extra"]
    core = cost["core_arithmetic"] + cost["core_extra"]
    return dict(
        model=name,
        task=task if name in DAS else None,
        preprocessing=pre,
        per_vector=core,
        reuse=reuse,
        equivalent_flops=core + pre / reuse,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    p.add_argument("--task", type=int, choices=range(24), default=10)
    p.add_argument("--reuse", type=int, default=64)
    p.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.reuse < 1:
        p.error("--reuse must be positive")
    rows = [count(n, a.task, a.reuse, a.weights / (n + ".pt")) for n in a.models]
    result = json.dumps(rows, indent=2)
    print(result)
    if a.output:
        a.output.write_text(result + "\n")
