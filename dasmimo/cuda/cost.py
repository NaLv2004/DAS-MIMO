import ast
import math
import re


def matching(text, pos, left="{", right="}"):
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == left:
            depth += 1
        elif text[i] == right:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced source")


def expression_cost(expression):
    """Count scalar floating expressions, folding literal-only arithmetic."""
    expression = expression.strip()
    # Every conditional floating assignment in these generated DAG kernels
    # uses the active-vector guard. Pure selector expressions only copy data.
    if expression.startswith("active ?"):
        expression = expression[len("active ?") :].rsplit(":", 1)[0].strip()
    elif "?" in expression:
        if not any(fn in expression for fn in ("fmaf", "fabsf", "pam4_")):
            return 0, 0
        raise ValueError("Unsupported arithmetic conditional: " + expression[:100])
    expression = re.sub(r"\[[^\]]*\]", "", expression)
    expression = re.sub(r"(?<=\d)[fF]\b", "", expression)
    expression = re.sub(
        r"\b([A-Za-z_]\w*)(?:\.\w+)+", lambda m: m[0].replace(".", "_"), expression
    )
    expression = expression.replace("{", "(").replace("}", ")")
    expression = expression.replace("reinterpret_cast<const float4*>", "cast")
    tree = ast.parse(" ".join(expression.split()), mode="eval").body

    def visit(node):
        if isinstance(node, ast.Constant):
            return 0, 0, node.value
        if isinstance(node, ast.Name):
            return 0, 0, None
        if isinstance(node, ast.UnaryOp):
            a, b, v = visit(node.operand)
            return a, b, -v if v is not None and isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b, x = visit(node.left)
            c, d, y = visit(node.right)
            if x is not None and y is not None:
                try:
                    value = {
                        ast.Add: lambda: x + y,
                        ast.Sub: lambda: x - y,
                        ast.Mult: lambda: x * y,
                        ast.Div: lambda: x / y,
                    }[type(node.op)]()
                    return a + c, b + d, value
                except (ZeroDivisionError, KeyError):
                    pass
            return a + c + 1, b + d, None
        if isinstance(node, ast.Call):
            args = [visit(arg) for arg in node.args]
            a, b = sum(v[0] for v in args), sum(v[1] for v in args)
            fn = node.func.id
            if fn in {"Pair", "Quad", "Oct", "pzero", "qzero", "ozero", "make_float4"}:
                return a, b, None
            if fn.startswith("__shfl"):
                return a, b, None
            if fn == "fmaf":
                return a + 2, b, None
            if fn == "pam4_denoise_invvar":
                # Stable two-exponential mean, with literal products folded.
                return a + 20, b + 8, None
            if fn == "pam4_denoise_tanh":
                # Fixed coefficients are folded at code generation.
                return a + 11, b + 1, None
            if fn == "softplus_stable":
                return a + 1, b + 4, None
            if fn in {
                "fabsf",
                "fminf",
                "fmaxf",
                "sqrtf",
                "rsqrtf",
                "expf",
                "__expf",
                "log1pf",
                "tanhf",
                "copysignf",
            }:
                return a, b + 1, None
            raise ValueError("Unpriced floating helper: " + fn)
        raise ValueError("Unrecognized expression: " + ast.dump(node))

    a, b, _ = visit(tree)
    return a, b


def dag_cost(text, task):
    match = re.search(
        r"__global__[^\n]*void dag_fixed_complex_pack2_task_" + f"{task:02d}" + r"\(",
        text,
    )
    start = text.index("{", match.end())
    body = text[start + 1 : matching(text, start)]
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    # Per-vector work begins after the persistent scheduling loop.
    body = body[body.index("Pair y0 =") :]
    loops = []
    constants = {"NC": 8, "MC": 16, "NR": 16, "MR": 32}
    pattern = r"for\s*\(int\s+(\w+)\s*=\s*(\d+);\s*\1\s*<\s*(\w+);[^)]*\)"
    for match in re.finditer(pattern, body):
        lo = int(match[2])
        hi = constants.get(match[3], int(match[3]) if match[3].isdigit() else None)
        if hi is None:
            raise ValueError("Nonconstant arithmetic loop")
        first = match.end()
        while body[first].isspace():
            first += 1
        end = matching(body, first) if body[first] == "{" else body.index(";", first)
        loops.append((first, end, hi - lo))
    assignments = re.compile(
        r"(?:(?:const\s+)?(float|Pair|Quad|Oct)\s+)?"
        r"([A-Za-z_]\w*(?:\.(?:lo|hi|r|i|x|y|z|w))*(?:\[[^\]]*\])?)"
        r"\s*(=|\+=|-=|\*=|/=)\s*([^;]+);",
        re.S,
    )
    float_names = set()
    arithmetic = extra = 0
    entries = []
    for match in assignments.finditer(body):
        kind, lhs, op, rhs = match.groups()
        if kind:
            float_names.add(lhs)
        if not (kind or "." in lhs or lhs in float_names or lhs.startswith("output[")):
            continue
        if lhs in {"y_base"} or "reinterpret_cast" in rhs:
            continue
        a, b = expression_cost(rhs)
        if op != "=":
            a += 1
        multiplier = 4  # Four active lanes own the 16 real coordinates.
        for begin, end, count in loops:
            if begin <= match.start() < end:
                multiplier *= count
        arithmetic += a * multiplier
        extra += b * multiplier
        if a or b:
            entries.append(
                {
                    "lhs": lhs,
                    "arithmetic": a * multiplier,
                    "equivalent_extra": b * multiplier,
                }
            )
    # grouped_dag_preprocess_parallel_kernel, exact 8x8 complex inverse.
    n, m = 8, 16
    chol = sum(8 * c + 2 + (2 if r == c else 2) for r in range(n) for c in range(r + 1))
    solves = n * 2 * (8 * n * (n - 1) // 2 + 2 * n)
    pre = {
        "full_complex_Gram": 8 * m * n * n,
        "norm_scale_and_regularization": n * (2 * n + 1) + 1,
        "normalized_Cholesky": chol,
        "inverse_triangular_solves": solves,
        "H_scaling": 2 * m * n,
        "A_scaling_and_D_inverse": 2 * n * n + 4 * n,
        "noise_scaling": 1,
    }
    pre_extra = 2 * n * n + n + 1 + 1 + 2 * n
    return {
        "pre_arithmetic": sum(pre.values()),
        "pre_extra": pre_extra,
        "core_arithmetic": arithmetic,
        "core_extra": extra,
        "pre_breakdown": pre,
        "core_statements": entries,
        "source": "generated DAG",
        "task": task,
    }


def grouped_admm_cost():
    """Count prepare<true> and admm<Threads> in learned.cu."""
    tx, rx, layers = 8, 16, 10
    n = 2 * tx
    triangle = n * (n - 1) // 2
    pre = {
        "Hermitian_Gram": rx * (4 * tx + 8 * (tx * (tx - 1) // 2)),
        "trace_jitter_and_diagonal_shift": tx - 1 + 2 + n,
        "Cholesky_column_scaling": triangle,
        "Cholesky_lower_triangle_updates": 2 * sum(j * (j + 1) // 2 for j in range(n)),
        "two_triangular_solves_for_each_inverse_column": n * 2 * (2 * triangle + n),
    }
    steps = layers - 1
    core = {
        "matched_filter": 2 * n * (2 * rx),
        "initial_inverse_times_matched_filter": 2 * n * n,
        "Gram_times_current_estimate": steps * 2 * n * n,
        "PAM_projection_theta_shift_sum_scale": layers * n * 6,
        "matched_residual": steps * n,
        "dual_update_first_and_remaining": n * (1 + 3 * (steps - 1)),
        "dual_extrapolation": steps * n * 3,
        "corrected_residual_and_symbol_update": steps * n * 4,
    }
    return {
        "source": "learned.cu",
        "iterations": layers,
        "pre_breakdown": pre,
        "pre_arithmetic": sum(pre.values()),
        "pre_extra": n + 1,  # Reciprocal square roots and trace clamp.
        "core_breakdown": core,
        "core_arithmetic": sum(core.values()),
        "core_extra": 3 * n * layers,  # Three tanh evaluations per real symbol.
    }


def bitonic_comparators(size: int) -> int:
    if size < 1 or size & (size - 1):
        raise ValueError("bitonic model requires a power-of-two size")
    stages = int(math.log2(size))
    return size // 2 * stages * (stages + 1) // 2


def baseline_costs():
    n, nc, nr = 16, 8, 16
    result = {}
    # Grouped AMP uses a compact Gram recurrence, not two H products/iteration.
    for iterations in (8, 15):
        updates = iterations - 1
        pre = 8 * nr * nc * nc + 2 * nr * nc + 2 * nc * nc + 1
        step = (
            2 * nc
            + 2 * nc
            + n * 31
            + nc
            + 3 * nc
            + 3
            + 2 * nc
            + 8 * nc * nc
            + 6 * nc
            + 10 * nc
        )
        result[f"damped_io_lama_t{iterations}"] = {
            "pre_arithmetic": pre,
            "pre_extra": 1,
            "core_arithmetic": 8 * nr * nc + 2 * nr * nc + nc + updates * step + 2 * nc,
            "core_extra": nc + updates * (n * 10 + nc),
            "updates": updates,
            "per_update_arithmetic": step,
            "source": "amp_ep.cu",
        }
    # EP: source-level 16-real-coordinate Cholesky and selected inverse diagonal.
    factor = 2 * n * sum(range(n)) + 3 * n + 2 * (n * (n - 1) // 2)
    solve = n * n + 4 * n * sum(range(n)) + 2 * n
    inv_lower = 2 * sum(r * (r + 1) // 2 for r in range(n)) + n * (n - 1) // 2
    inv_diag = n * (n + 1)
    site = n * (6 + 37 + 6 + 4)
    ep_step = factor + solve + inv_lower + inv_diag + site
    ep_pre = (
        8 * nr * nc * nc
        + 2 * nr * nc
        + 2 * nc * nc
        + nc
        + sum(8 * c for r in range(nc) for c in range(r + 1))
        + 3 * nc
        + 4 * nc * (nc - 1) // 2
        + nc * (16 * nc * sum(range(nc)) + 4 * nc)
    )
    result["ep_t7"] = {
        "pre_arithmetic": ep_pre,
        "pre_extra": nc + 2 * nc,
        "core_arithmetic": 8 * nr * nc + 8 * n * nc + 7 * ep_step,
        "core_extra": 7 * (n * 12 + 2 * n),
        "per_iteration_arithmetic": ep_step,
        "source": "amp_ep.cu",
    }
    # K=8: all 15 pruning levels execute a full 32-lane bitonic network,
    # including the padded 16-candidate second level.
    levels = []
    for depth in range(n):
        parents = min(8, 4**depth)
        candidates, survivors = 4 * parents, min(8, 4 * parents)
        levels.append(
            {
                "depth": depth,
                "arithmetic": parents * (depth + 1) + 3 * candidates,
                "sort_equivalents": 0
                if candidates <= 8
                else 5 * bitonic_comparators(32),
                "path_equivalents": 3 * parents * depth
                + 4 * candidates
                + 2 * survivors,
            }
        )
    chol_real = sum(
        2 * c + 2 if r > c else 2 * c + 2 for r in range(n) for c in range(r + 1)
    )
    result["kbest_k8_maxlog"] = {
        "pre_arithmetic": 8 * nr * (nc * (nc + 1) // 2)
        + chol_real
        + 4 * n * (n + 1) // 2,
        "pre_extra": 2 * n,
        "core_arithmetic": 8 * nr * nc
        + 2 * n * sum(range(n))
        + n
        + sum(v["arithmetic"] for v in levels),
        "core_extra": sum(v["sort_equivalents"] + v["path_equivalents"] for v in levels)
        + 3 * n,
        "levels": levels,
        "source": "kbest.cu",
        "sort_convention": "one comparison plus four scalar record moves per compare-exchange",
        "output_boundary": "symbol output",
    }
    # Fused ChannelNet: 10 RX MLPs, 9 TX MLPs and one 10-50-50-4 head.
    core = (
        10 * 32 * 400
        + 9 * 16 * 400
        + 10 * 8 * 16 * 8 * 10
        + 9 * 8 * 16 * 8 * 10
        + 9 * 16 * 10
        + 9 * 32 * 10
        + 16 * (2 * 10 * 50 + 2 * 50 * 50 + 2 * 50 * 4)
        + 16 * (4 * 4 + 1)
    )
    result["channelnet_l10"] = {
        "pre_arithmetic": 0,
        "pre_extra": 0,
        "core_arithmetic": core,
        "core_extra": 10 * 32 * 20 + 9 * 16 * 20 + 16 * 100 + 16 * 7,
        "source": "channelnet",
        "pre_note": "Channel staging uses memory copies only, not floating arithmetic.",
    }
    hidden, state, axes = 128, 64, 16
    layer = (
        2 * n * n
        + 4 * n
        + 2 * (n + state) * hidden
        + 2 * hidden * (128 + state)
        + 3 * (128 + state)
        + 2 * axes * 16
    )
    result["detnet_l10"] = dict(
        pre_arithmetic=2 * n * n * 2 * nr,
        pre_extra=0,
        core_arithmetic=2 * n * 2 * nr + 9 * layer,
        core_extra=9 * hidden,
    )
    result["admm_t10"] = grouped_admm_cost()
    ad = result["admm_t10"]
    result["mmse"] = dict(
        pre_arithmetic=ad["pre_arithmetic"] - (8 - 1 + 2),
        pre_extra=16,
        core_arithmetic=2 * 16 * 32 + 2 * 16 * 16,
        core_extra=0,
    )
    ep = dict(result["ep_t7"])
    ep["core_arithmetic"] -= 4 * ep_step
    ep["core_extra"] = 3 * (n * 12 + 2 * n)
    result["ep_t3"] = ep
    return result
