import contextvars
import itertools
import math
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

_ACTIVE_CODE_ARCH_LATENT = contextvars.ContextVar(
    "active_code_architecture_latent", default=None
)


class CodeArchitectureParametrization(nn.Module):
    """Bounded K-template adjustment for one existing DAG parameter tensor."""

    def __init__(self, original, rank, max_delta, init_std):
        super().__init__()
        if rank <= 0:
            raise ValueError("CODE_ARCH_SHARING_K must be positive")
        if max_delta <= 0:
            raise ValueError("Code-architecture maximum deltas must be positive")
        self.rank = int(rank)
        self.max_delta = float(max_delta)
        basis = torch.empty(
            (self.rank,) + tuple(original.shape),
            device=original.device,
            dtype=original.dtype,
        )
        nn.init.normal_(basis, mean=0.0, std=float(init_std))
        self.register_buffer("finite_mask", (original.detach() > -1e8), persistent=True)
        self.basis = nn.Parameter(basis)

    def forward(self, original):
        latent = _ACTIVE_CODE_ARCH_LATENT.get()
        if latent is None:
            return original
        if latent.dim() == 2:
            if latent.shape[0] != 1:
                raise ValueError(
                    "A code-conditioned DAG batch must contain one LDPC task"
                )
            latent = latent[0]
        if latent.dim() != 1 or latent.numel() != self.rank:
            raise ValueError(
                f"Expected a [{self.rank}] code latent, got {tuple(latent.shape)}"
            )
        latent = latent.to(device=self.basis.device, dtype=self.basis.dtype)
        mixed = torch.tensordot(latent, self.basis, dims=([0], [0]))
        delta = self.max_delta * torch.tanh(mixed)
        return torch.where(self.finite_mask, original + delta, original)

    def group_lasso(self):
        return torch.sqrt(
            self.basis.square().sum(dim=0) + torch.finfo(self.basis.dtype).eps
        ).mean()

    def delta_l2(self, latent):
        latent = latent.to(device=self.basis.device, dtype=self.basis.dtype)
        mixed = torch.tensordot(latent, self.basis, dims=([0], [0]))
        delta = self.max_delta * torch.tanh(mixed)
        finite = delta[self.finite_mask]
        return finite.square().mean() if finite.numel() else delta.new_zeros(())


class CodeArchitectureEncoder(nn.Module):
    """Map four inexpensive code descriptors to K global template weights."""

    def __init__(self, cfg):
        super().__init__()
        descriptor_dim = int(getattr(cfg, "CODE_ARCH_DESCRIPTOR_DIM", 4))
        hidden = int(getattr(cfg, "CODE_ARCH_ENCODER_HIDDEN", 16))
        rank = int(getattr(cfg, "CODE_ARCH_SHARING_K", 8))
        if descriptor_dim <= 0 or hidden <= 0 or rank <= 0:
            raise ValueError("Code-architecture encoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, rank, bias=False),
        )
        self.rank = rank

    def forward(self, descriptor):
        descriptor = torch.as_tensor(
            descriptor,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        if descriptor.dim() == 1:
            descriptor = descriptor.unsqueeze(0)
        if descriptor.dim() != 2:
            raise ValueError("code_descriptor must be one- or two-dimensional")
        latent = torch.tanh(self.network(descriptor))
        return latent - latent.mean(dim=-1, keepdim=True)


class DataType(Enum):
    SCALAR = 0
    VEC_N = 1
    VEC_M = 2


class SemanticTag(Enum):
    UNKNOWN = 0
    SIGNAL = 1
    RESIDUAL = 2
    MOMENTUM = 3
    GENERIC = 4


class OpRegistry:
    def __init__(self):
        self.ops = []

    def register(
        self, name, out_type, in_types, func, is_learnable=False, semantic_rule=None
    ):
        self.ops.append(
            {
                "name": name,
                "func": func,
                "out": out_type,
                "in": in_types,
                "learnable": is_learnable,
                "semantic_rule": semantic_rule,
            }
        )


OPS = OpRegistry()


def get_shape(type_enum, Nt, Nr):
    if type_enum == DataType.SCALAR:
        return (1, 1)
    if type_enum == DataType.VEC_N:
        return (Nt, 1)
    if type_enum == DataType.VEC_M:
        return (Nr, 1)
    return (1, 1)


def rule_pass(inputs):
    return inputs[0]


def rule_linear_preserve(inputs):
    s1, s2 = inputs[0], inputs[1]
    if s1 == SemanticTag.SIGNAL or s2 == SemanticTag.SIGNAL:
        return SemanticTag.SIGNAL
    if s1 == SemanticTag.MOMENTUM or s2 == SemanticTag.MOMENTUM:
        return SemanticTag.MOMENTUM
    if s1 == SemanticTag.RESIDUAL or s2 == SemanticTag.RESIDUAL:
        return SemanticTag.RESIDUAL
    return SemanticTag.GENERIC


def rule_mul_H(inputs):
    return SemanticTag.RESIDUAL


def rule_mul_HT(inputs):
    return SemanticTag.MOMENTUM


def rule_nonlinear_check(inputs):
    if inputs[0] in [SemanticTag.SIGNAL, SemanticTag.MOMENTUM]:
        return inputs[0]
    return SemanticTag.UNKNOWN


def rule_gradient_step(inputs):
    if inputs[0] == SemanticTag.SIGNAL and inputs[1] == SemanticTag.RESIDUAL:
        return SemanticTag.SIGNAL
    return SemanticTag.UNKNOWN


OPS.register(
    "Pass_N",
    DataType.VEC_N,
    [DataType.VEC_N],
    lambda x, env: x,
    semantic_rule=rule_pass,
)
OPS.register(
    "Pass_M",
    DataType.VEC_M,
    [DataType.VEC_M],
    lambda x, env: x,
    semantic_rule=rule_pass,
)
OPS.register(
    "Pass_S",
    DataType.SCALAR,
    [DataType.SCALAR],
    lambda x, env: x,
    semantic_rule=rule_pass,
)

OPS.register(
    "Mul_A",
    DataType.VEC_N,
    [DataType.VEC_N],
    lambda x, env: torch.bmm(env["A"], x),
    semantic_rule=rule_pass,
)
OPS.register(
    "Mul_H",
    DataType.VEC_M,
    [DataType.VEC_N],
    lambda x, env: torch.bmm(env["H"], x),
    semantic_rule=rule_mul_H,
)
OPS.register(
    "Mul_HT",
    DataType.VEC_N,
    [DataType.VEC_M],
    lambda x, env: torch.bmm(env["H_T"], x),
    semantic_rule=rule_mul_HT,
)
OPS.register(
    "Mul_Dinv",
    DataType.VEC_N,
    [DataType.VEC_N],
    lambda x, env: torch.bmm(env["D_inv"], x),
    semantic_rule=rule_pass,
)

OPS.register(
    "Mul_Ainv",
    DataType.VEC_N,
    [DataType.VEC_N],
    lambda x, env: torch.bmm(env["A_inv"], x),
    semantic_rule=rule_pass,
)

OPS.register(
    "Add_N",
    DataType.VEC_N,
    [DataType.VEC_N, DataType.VEC_N],
    lambda x, y, env: x + y,
    semantic_rule=rule_linear_preserve,
)
OPS.register(
    "Sub_N",
    DataType.VEC_N,
    [DataType.VEC_N, DataType.VEC_N],
    lambda x, y, env: x - y,
    semantic_rule=rule_linear_preserve,
)


def _learnable_param_component(env, param_idx, component, ref_tensor=None):
    params = env["learnable_params"][param_idx]
    value = params[..., component]
    if value.dim() == 0:
        return value.view(1, 1, 1)
    dtype = ref_tensor.dtype if ref_tensor is not None else value.dtype
    return value.view(-1, 1, 1).to(dtype=dtype)


def op_mix_n(x, y, env, param_idx):
    alpha = torch.sigmoid(_learnable_param_component(env, param_idx, 0, x))
    return (alpha * x + (1.0 - alpha) * y).to(x.dtype)


OPS.register(
    "Mix_N",
    DataType.VEC_N,
    [DataType.VEC_N, DataType.VEC_N],
    op_mix_n,
    is_learnable=True,
    semantic_rule=rule_linear_preserve,
)


def op_norm2_m_scaled(x, env, param_idx):
    scale = F.softplus(_learnable_param_component(env, param_idx, 0, x))
    return (torch.mean(x**2, dim=1, keepdim=True) * scale).to(x.dtype)


OPS.register(
    "Norm2_M_Scaled",
    DataType.SCALAR,
    [DataType.VEC_M],
    op_norm2_m_scaled,
    is_learnable=True,
    semantic_rule=lambda x: SemanticTag.GENERIC,
)
OPS.register(
    "Norm2_M",
    DataType.SCALAR,
    [DataType.VEC_M],
    lambda x, env: torch.mean(x**2, dim=1, keepdim=True),
    semantic_rule=lambda x: SemanticTag.GENERIC,
)
OPS.register(
    "Mean_Abs_M",
    DataType.SCALAR,
    [DataType.VEC_M],
    lambda x, env: torch.mean(torch.abs(x), dim=1, keepdim=True),
    semantic_rule=lambda x: SemanticTag.GENERIC,
)
OPS.register(
    "Add_S",
    DataType.SCALAR,
    [DataType.SCALAR, DataType.SCALAR],
    lambda x, y, env: x + y,
    semantic_rule=lambda x: SemanticTag.GENERIC,
)
OPS.register(
    "Div_S",
    DataType.SCALAR,
    [DataType.SCALAR, DataType.SCALAR],
    lambda x, y, env: x / (torch.abs(y) + 1e-6),
    semantic_rule=lambda x: SemanticTag.GENERIC,
)


def op_cut(x, env, param_idx):
    lam = _learnable_param_component(env, param_idx, 0, x).abs()
    return torch.minimum(torch.maximum(x, -lam), lam)


def soft_16qam_slicer_dynamic(x, var_scalar, env, param_idx):
    params = env["learnable_params"][param_idx]
    if params.dim() == 1:
        p0 = params[0].view(1, 1, 1)
        p1 = params[1].view(1, 1, 1)
    else:
        p0 = params[:, 0].view(-1, 1, 1)
        p1 = params[:, 1].view(-1, 1, 1)
    variance = F.softplus(p0 * var_scalar + p1) + 1e-4
    precision = 1.0 / variance

    centers = env.get("qam_centers")
    if centers is None or centers.device != x.device or centers.dtype != x.dtype:
        scale = 1.0 / math.sqrt(10)
        centers = (
            torch.tensor([-3.0, -1.0, 1.0, 3.0], device=x.device, dtype=x.dtype) * scale
        )

    x_expanded = x.unsqueeze(-1)
    centers_expanded = centers.view(1, 1, 1, -1)
    dist_sq = (x_expanded - centers_expanded) ** 2

    prec_expanded = precision.unsqueeze(-1)
    probs = torch.softmax(-prec_expanded * dist_sq, dim=-1)
    output = torch.sum(probs * centers_expanded, dim=-1)
    return output.to(x.dtype)


OPS.register(
    "Cut_N",
    DataType.VEC_N,
    [DataType.VEC_N],
    op_cut,
    is_learnable=True,
    semantic_rule=rule_nonlinear_check,
)

OPS.register(
    "SoftSlicer",
    DataType.VEC_N,
    [DataType.VEC_N, DataType.SCALAR],
    soft_16qam_slicer_dynamic,
    is_learnable=True,
    semantic_rule=lambda x: x[0] if x[0] == SemanticTag.SIGNAL else SemanticTag.UNKNOWN,
)


def gumbel_softmax_ste_compact(
    logits,
    source_indices,
    temperature=1.0,
    hard=False,
    is_training=False,
    noise_scale=0.02,
):
    if is_training and float(noise_scale) > 0:
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = torch.index_select(
            logits + gumbel_noise * float(noise_scale), 0, source_indices
        )
    else:
        y = torch.index_select(logits, 0, source_indices)

    y_soft = F.softmax(y / temperature, dim=-1)

    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret if ret.dtype == logits.dtype else ret.to(logits.dtype)


class DAGNode(nn.Module):
    def __init__(self, node_idx, prev_nodes_info, cfg):
        super().__init__()
        self.node_idx = node_idx
        self.cfg = cfg
        self.ops_registry = OPS.ops
        self.prev_info = prev_nodes_info

        self.op_weights = nn.ParameterDict()
        self.op_select_logits = nn.ParameterDict()
        self.valid_ops_map = {}
        self.edge_weights = nn.ParameterDict()
        self.internal_params = nn.ParameterList()
        self.param_map = {}
        self.output_semantics = {}
        self._build_params_and_infer_semantics()
        self._build_compiled_forward_plan()

    def _build_params_and_infer_semantics(self):
        possible_out_types = set()
        for out_t in DataType:
            valid_ops_indices = []
            derived_semantics_for_this_type = set()

            for i, op in enumerate(self.ops_registry):
                if op["out"] != out_t:
                    continue
                if op["name"] == "Mul_Ainv" and not bool(
                    getattr(self.cfg, "ENABLE_MUL_AINV", False)
                ):
                    continue

                req_in_types = op["in"]
                arg_possibilities = []
                for req_type in req_in_types:
                    tags_available = set()
                    for prev_idx in range(self.node_idx):
                        prev_node_meta = self.prev_info[prev_idx]
                        if req_type in prev_node_meta["types"]:
                            tags = prev_node_meta["semantics"].get(
                                req_type, {SemanticTag.GENERIC}
                            )
                            tags_available.update(tags)
                    arg_possibilities.append(tags_available)

                if req_in_types and not all(arg_possibilities):
                    continue

                is_semantically_valid = False
                for inputs_tags in itertools.product(*arg_possibilities):
                    res_tag = op["semantic_rule"](inputs_tags)
                    if res_tag != SemanticTag.UNKNOWN:
                        is_semantically_valid = True
                        derived_semantics_for_this_type.add(res_tag)

                if is_semantically_valid:
                    valid_ops_indices.append(i)

            if not valid_ops_indices:
                continue

            key_name = out_t.name
            self.valid_ops_map[key_name] = valid_ops_indices

            self.op_weights[key_name] = nn.Parameter(
                torch.randn(len(valid_ops_indices)) * 0.1
            ).to(self.cfg.dtype)
            self.op_select_logits[key_name] = nn.Parameter(
                torch.zeros(len(valid_ops_indices))
            ).to(self.cfg.dtype)
            self.output_semantics[out_t] = derived_semantics_for_this_type
            possible_out_types.add(out_t)

            for op_idx in valid_ops_indices:
                op_data = self.ops_registry[op_idx]
                if op_data["learnable"]:
                    p = nn.Parameter(torch.randn(2) * 0.1).to(self.cfg.dtype)
                    self.internal_params.append(p)
                    self.param_map[op_idx] = len(self.internal_params) - 1

                for arg_idx, req_type in enumerate(op_data["in"]):
                    key = f"op{op_idx}_arg{arg_idx}"
                    is_sensitive_op = "Slicer" in op_data["name"] or op_data[
                        "name"
                    ] in ["Cut_N", "LeakyReLU_N"]

                    w = torch.randn(self.node_idx, dtype=self.cfg.dtype) * 0.1
                    valid_sources_count = 0

                    for prev_i in range(self.node_idx):
                        prev_meta = self.prev_info[prev_i]
                        if req_type not in prev_meta["types"]:
                            w[prev_i] = -1e9
                            continue

                        if is_sensitive_op and req_type in [
                            DataType.VEC_N,
                            DataType.VEC_M,
                        ]:
                            tags = prev_meta["semantics"].get(req_type, set())
                            if not (
                                SemanticTag.SIGNAL in tags
                                or SemanticTag.MOMENTUM in tags
                            ):
                                w[prev_i] = -1e9
                                continue

                        valid_sources_count += 1

                    self.edge_weights[key] = nn.Parameter(w)

    def _build_compiled_forward_plan(self):
        self._compiled_ops = {}
        buffer_idx = 0
        for out_t_name, op_indices in self.valid_ops_map.items():
            compiled = []
            for local_idx, op_idx in enumerate(op_indices):
                op_data = self.ops_registry[op_idx]
                args = []
                for arg_idx, req_type in enumerate(op_data["in"]):
                    edge_key = f"op{op_idx}_arg{arg_idx}"
                    raw_w = self.edge_weights[edge_key].detach()
                    source_indices = tuple(
                        i for i in range(self.node_idx) if raw_w[i].item() > -1e8
                    )
                    buffer_name = f"_compiled_src_{buffer_idx}"
                    self.register_buffer(
                        buffer_name,
                        torch.tensor(source_indices, dtype=torch.long),
                        persistent=False,
                    )
                    buffer_idx += 1
                    args.append((req_type, edge_key, source_indices, buffer_name))
                compiled.append((local_idx, op_idx, op_data, args))
            self._compiled_ops[out_t_name] = compiled

    def forward(self, slots, env, use_threshold, threshold):
        outputs = {}
        env["learnable_params"] = self.internal_params
        for kind, operations in self._compiled_ops.items():
            weights = self.op_weights[kind]
            total = None
            for index, op_index, op, arguments in operations:
                weight = weights[index]
                if use_threshold and weight.detach().abs() < threshold:
                    continue
                operands = []
                for dtype, key, candidates, buffer in arguments:
                    indices = [i for i in candidates if dtype in slots[i]]
                    if not indices:
                        break
                    logits = self.edge_weights[key]
                    source = (
                        getattr(self, buffer)
                        if len(indices) == len(candidates)
                        else logits.new_tensor(indices, dtype=torch.long)
                    )
                    if not self.training:
                        chosen = indices[int(logits[source].argmax())]
                        operands.append(slots[chosen][dtype])
                    else:
                        coeff = gumbel_softmax_ste_compact(
                            logits,
                            source,
                            env["tau"],
                            True,
                            True,
                            self.cfg.INPUT_SOURCE_GUMBEL_NOISE_SCALE,
                        )
                        values = torch.stack([slots[i][dtype] for i in indices])
                        operands.append((values * coeff.view(-1, 1, 1, 1)).sum(0))
                else:
                    args = [*operands, env]
                    if op["learnable"]:
                        args.append(self.param_map[op_index])
                    term = weight * op["func"](*args)
                    total = term if total is None else total + term
            if total is not None:
                outputs[DataType[kind]] = total
        return outputs


class DAGLayer(nn.Module):
    def __init__(
        self, num_nodes, input_config, output_target_type, cfg, extra_output_types=None
    ):
        super().__init__()
        self.output_target_type = output_target_type
        self.extra_output_types = tuple(extra_output_types or ())
        self.output_types = tuple(
            dict.fromkeys((self.output_target_type,) + self.extra_output_types)
        )
        self.nodes = nn.ModuleList()
        self.node_info = []
        self.compute_output_indices = []
        self.slot_producers = {}

        for inp in input_config:
            info = {"types": set(), "semantics": {}, "producer_kind": "input"}
            for dtype, tags in inp.items():
                info["types"].add(dtype)
                if isinstance(tags, SemanticTag):
                    tags = {tags}
                info["semantics"][dtype] = tags
            self.node_info.append(info)

        self.start_node_idx = len(self.node_info)

        for i in range(num_nodes):
            node = DAGNode(len(self.node_info), self.node_info, cfg)
            self.nodes.append(node)

            self.compute_output_indices.append(len(self.node_info))
            self.slot_producers[len(self.node_info)] = {
                "kind": "combined",
                "node_local_idx": i,
            }
            info = {"types": set(), "semantics": {}, "producer_kind": "combined"}
            for out_t, tags in node.output_semantics.items():
                info["types"].add(out_t)
                info["semantics"][out_t] = tags
            self.node_info.append(info)

    def _last_output_or_zero(self, node_outputs, target_type, env):
        final_out = None
        search_indices = list(reversed(self.compute_output_indices))
        search_indices.extend(range(self.start_node_idx - 1, -1, -1))
        for i in search_indices:
            if target_type in node_outputs[i]:
                final_out = node_outputs[i][target_type]
                break

        if final_out is None:
            shape = get_shape(target_type, env["Nt"], env["Nr"])
            B = env["A"].shape[0]
            final_out = torch.zeros(
                B, shape[0], shape[1], device=env["A"].device, dtype=env["A"].dtype
            )
        return final_out

    def forward(self, inputs, env, use_threshold, threshold):
        slots = inputs.copy()
        for node in self.nodes:
            slots.append(node(slots, env, use_threshold, threshold))
        return {t: self._last_output_or_zero(slots, t, env) for t in self.output_types}


def _m_memory_input_config(count):
    return [{DataType.VEC_M: {SemanticTag.GENERIC}} for _ in range(count)]


def _m_memory_inputs(memory_outputs):
    return [{DataType.VEC_M: value} for value in memory_outputs]


def _n_memory_input_config(count):
    return [{DataType.VEC_N: {SemanticTag.GENERIC}} for _ in range(count)]


def _n_memory_inputs(memory_outputs):
    return [{DataType.VEC_N: value} for value in memory_outputs]


def _build_iteration_layers(cfg):
    phis, psis = nn.ModuleList(), nn.ModuleList()
    for k in range(cfg.N_ITER):
        for offset, layers, inputs in [
            (
                0,
                phis,
                [
                    {DataType.VEC_M: {SemanticTag.RESIDUAL}},
                    {DataType.VEC_M: {SemanticTag.RESIDUAL}},
                    {DataType.VEC_N: {SemanticTag.SIGNAL}},
                    {DataType.SCALAR: {SemanticTag.GENERIC}},
                ],
            ),
            (
                1,
                psis,
                [
                    {DataType.VEC_N: {SemanticTag.MOMENTUM}},
                    {DataType.VEC_N: {SemanticTag.MOMENTUM}},
                    {DataType.SCALAR: {SemanticTag.GENERIC}},
                ],
            ),
        ]:
            count = 2 * k + offset
            inputs.extend(_n_memory_input_config(count))
            inputs.extend(_m_memory_input_config(count))
            layers.append(
                DAGLayer(cfg.DAG_NODES, inputs, DataType.VEC_N, cfg, (DataType.VEC_M,))
            )
    return phis, psis


class DASDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.phis, self.psis = _build_iteration_layers(cfg)
        self.register_buffer("step_counter", torch.tensor(0.0))
        self.code_arch_encoder = CodeArchitectureEncoder(cfg)
        self._code_arch_parametrizations = []
        self._last_code_architecture_latent = None
        self._install_code_architecture_parametrizations()

    def _code_arch_parameter_kind(self, full_name):
        if ".state_edge_weights." in full_name or ".edge_weights." in full_name:
            return "edge"
        if ".state_op_select_logits." in full_name or ".op_select_logits." in full_name:
            return "op_logit"
        if ".state_op_weights." in full_name or ".op_weights." in full_name:
            return "op_weight"
        if ".internal_params." in full_name and bool(
            getattr(self.cfg, "CODE_ARCH_CONDITION_INTERNAL_PARAMS", True)
        ):
            return "internal"
        return None

    def _code_arch_max_delta(self, kind):
        fields = {
            "edge": "CODE_ARCH_EDGE_MAX_DELTA",
            "op_logit": "CODE_ARCH_OP_LOGIT_MAX_DELTA",
            "op_weight": "CODE_ARCH_OP_WEIGHT_MAX_DELTA",
            "internal": "CODE_ARCH_INTERNAL_MAX_DELTA",
        }
        return float(getattr(self.cfg, fields[kind]))

    def _install_code_architecture_parametrizations(self):
        """Attach one K-basis adjustment to every existing architecture tensor."""
        targets = []
        for module_name, module in self.named_modules():
            if module_name.startswith("code_arch_encoder"):
                continue
            for parameter_name, parameter in list(
                module.named_parameters(recurse=False)
            ):
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                kind = self._code_arch_parameter_kind(full_name)
                if kind is not None:
                    targets.append((module, parameter_name, parameter, kind))

        rank = int(getattr(self.cfg, "CODE_ARCH_SHARING_K", 8))
        init_std = float(getattr(self.cfg, "CODE_ARCH_BASIS_INIT_STD", 0.01))
        if init_std < 0:
            raise ValueError("CODE_ARCH_BASIS_INIT_STD must be non-negative")
        for module, parameter_name, parameter, kind in targets:
            adjustment = CodeArchitectureParametrization(
                parameter,
                rank=rank,
                max_delta=self._code_arch_max_delta(kind),
                init_std=init_std,
            )
            parametrize.register_parametrization(
                module, parameter_name, adjustment, unsafe=False
            )
            self._code_arch_parametrizations.append(adjustment)
        if not self._code_arch_parametrizations:
            raise RuntimeError("Code architecture sharing found no DAG parameters")
        self.register_buffer(
            "_code_arch_valid_flat",
            torch.cat(
                [
                    adjustment.finite_mask.reshape(-1)
                    for adjustment in self._code_arch_parametrizations
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "_code_arch_max_delta_flat",
            torch.cat(
                [
                    adjustment.basis.new_full(
                        (adjustment.basis[0].numel(),), adjustment.max_delta
                    )
                    for adjustment in self._code_arch_parametrizations
                ]
            ),
            persistent=False,
        )

    def _flat_code_architecture_basis(self):
        rank = int(getattr(self.cfg, "CODE_ARCH_SHARING_K", 8))
        return torch.cat(
            [
                adjustment.basis.reshape(rank, -1)
                for adjustment in self._code_arch_parametrizations
            ],
            dim=1,
        )

    def _activate_code_architecture(self, code_descriptor):
        if self.code_arch_encoder is None:
            _ACTIVE_CODE_ARCH_LATENT.set(None)
            self._last_code_architecture_latent = None
            return None
        if code_descriptor is None:
            code_descriptor = getattr(self.cfg, "CODE_ARCH_DEFAULT_DESCRIPTOR", None)
        if code_descriptor is None:
            raise ValueError(
                "CODE_ARCH_SHARING_ENABLED requires a code_descriptor for each task batch"
            )
        latent = self.code_arch_encoder(code_descriptor)
        if latent.shape[0] != 1:
            raise ValueError(
                "One detector forward may contain only one homogeneous LDPC task"
            )
        latent = latent[0]
        self._last_code_architecture_latent = latent
        _ACTIVE_CODE_ARCH_LATENT.set(latent)
        return latent

    def code_architecture_regularization(self, descriptor_bank=None):
        first = next(self.parameters())
        zero = first.new_zeros(())
        if self.code_arch_encoder is None:
            return zero, {
                "group_lasso": zero,
                "delta_l2": zero,
                "latent_orthogonal": zero,
            }
        if descriptor_bank is None:
            latent = self._last_code_architecture_latent
            if latent is None:
                raise RuntimeError("No code architecture has been activated")
            latents = latent.unsqueeze(0)
        else:
            latents = self.code_arch_encoder(descriptor_bank)

        basis = self._flat_code_architecture_basis()
        valid = self._code_arch_valid_flat
        valid_basis = basis[:, valid]
        group_lasso = torch.sqrt(
            valid_basis.square().sum(dim=0) + torch.finfo(valid_basis.dtype).eps
        ).mean()
        active_latent = self._last_code_architecture_latent
        if active_latent is None:
            active_latent = latents.mean(dim=0)
        mixed = active_latent.to(valid_basis).matmul(valid_basis)
        delta = self._code_arch_max_delta_flat[valid] * torch.tanh(mixed)
        delta_l2 = delta.square().mean()

        if latents.shape[0] > 1:
            centered = latents - latents.mean(dim=0, keepdim=True)
            covariance = centered.transpose(0, 1).matmul(centered) / float(
                latents.shape[0]
            )
            off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
            latent_orthogonal = off_diagonal.square().mean()
        else:
            latent_orthogonal = zero
        weighted = (
            float(getattr(self.cfg, "CODE_ARCH_GROUP_LASSO_WEIGHT", 0.0)) * group_lasso
            + float(getattr(self.cfg, "CODE_ARCH_DELTA_L2_WEIGHT", 0.0)) * delta_l2
            + float(getattr(self.cfg, "CODE_ARCH_LATENT_ORTHOGONAL_WEIGHT", 0.0))
            * latent_orthogonal
        )
        return weighted, {
            "group_lasso": group_lasso,
            "delta_l2": delta_l2,
            "latent_orthogonal": latent_orthogonal,
        }

    def detach_code_architecture_state(self):
        if self._last_code_architecture_latent is None:
            return
        self._last_code_architecture_latent = (
            self._last_code_architecture_latent.detach()
        )
        _ACTIVE_CODE_ARCH_LATENT.set(self._last_code_architecture_latent)

    def forward(
        self,
        H,
        y,
        A,
        s_factor,
        use_threshold=False,
        threshold=1e-3,
        code_descriptor=None,
    ):
        B, Nr, Nt = H.shape
        device = H.device

        self._activate_code_architecture(code_descriptor)
        tau = (
            max(0.1, 0.5 * math.exp(-5e-5 * self.step_counter.item()))
            if self.training
            else 0.1
        )

        sqrt_s = torch.sqrt(s_factor.view(B, 1, 1) + 1e-8).to(self.cfg.dtype)
        H_norm = H / sqrt_s
        H_T_norm = H.transpose(1, 2) / sqrt_s
        diag_A = torch.diagonal(A, dim1=-2, dim2=-1)
        D_inv = torch.diag_embed(1.0 / (diag_A + 1e-6)).to(self.cfg.dtype)

        qam_centers = torch.tensor(
            [-3.0, -1.0, 1.0, 3.0],
            device=device,
            dtype=self.cfg.dtype,
        ) * (1.0 / math.sqrt(10))

        gram_norm = torch.bmm(H_T_norm, H_norm)
        noise_variance = (
            torch.diagonal(A - gram_norm, dim1=-2, dim2=-1)
            .mean(dim=1, keepdim=True)
            .clamp_min(1e-8)
        )
        env = {
            "A": A,
            "H": H_norm,
            "H_T": H_T_norm,
            "y": y / sqrt_s,
            "D_inv": D_inv,
            "Nt": Nt,
            "Nr": Nr,
            "tau": tau,
            "qam_centers": qam_centers,
            "noise_variance": noise_variance,
            "gram": gram_norm,
            "cfg": self.cfg,
            "is_training": self.training,
        }
        if bool(getattr(self.cfg, "ENABLE_MUL_AINV", False)):
            env["A_inv"] = torch.linalg.inv(A)
        y_norm = y / sqrt_s

        x_k = torch.zeros(B, Nt, 1, device=device, dtype=self.cfg.dtype)
        x_km1 = x_k.clone()
        x_km2 = x_k.clone()
        res_prev = torch.zeros(B, Nr, 1, device=device, dtype=self.cfg.dtype)

        trajectory = []
        base_scalar = torch.ones(B, 1, 1, device=device, dtype=self.cfg.dtype)
        n_memory_outputs = []
        m_memory_outputs = []

        for k in range(self.cfg.N_ITER):
            res = y_norm - torch.bmm(H_norm, x_k)
            diff = x_k - x_km1
            diff_prev = x_km1 - x_km2

            phi_inputs = [
                {DataType.VEC_M: res},
                {DataType.VEC_M: res_prev},
                {DataType.VEC_N: x_k},
                {DataType.SCALAR: base_scalar},
            ]
            phi_inputs.extend(_n_memory_inputs(n_memory_outputs))
            phi_inputs.extend(_m_memory_inputs(m_memory_outputs))
            phi_outputs = self.phis[k](
                phi_inputs,
                env,
                use_threshold,
                threshold,
            )
            correction = phi_outputs[DataType.VEC_N]
            n_memory_outputs.append(correction)
            m_memory_outputs.append(phi_outputs[DataType.VEC_M])

            psi_inputs = [
                {DataType.VEC_N: diff},
                {DataType.VEC_N: diff_prev},
                {DataType.SCALAR: base_scalar},
            ]
            psi_inputs.extend(_n_memory_inputs(n_memory_outputs))
            psi_inputs.extend(_m_memory_inputs(m_memory_outputs))
            psi_outputs = self.psis[k](
                psi_inputs,
                env,
                use_threshold,
                threshold,
            )
            momentum = psi_outputs[DataType.VEC_N]
            n_memory_outputs.append(momentum)
            m_memory_outputs.append(psi_outputs[DataType.VEC_M])

            x_next = correction + momentum + x_k
            x_next = torch.clamp(x_next, -10, 10)

            res_prev = res
            x_km2 = x_km1
            x_km1 = x_k
            x_k = x_next
            trajectory.append(x_k)

        return trajectory
