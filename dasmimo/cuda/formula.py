from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union

import torch

from ..model import DASDetector, DataType


@dataclass(frozen=True)
class Zero:
    dtype: DataType


@dataclass(frozen=True)
class Constant:
    dtype: DataType
    value: float


@dataclass(frozen=True)
class Variable:
    dtype: DataType
    name: str


@dataclass(frozen=True)
class LinearCombination:
    dtype: DataType
    terms: Tuple[Tuple[float, "Expression"], ...]


@dataclass(frozen=True)
class Call:
    dtype: DataType
    name: str
    args: Tuple["Expression", ...]
    params: Tuple[float, ...] = ()
    registry: str = "ordinary"
    output_index: int = 0


Expression = Union[Zero, Constant, Variable, LinearCombination, Call]


LINEAR_UNARY_OPS = frozenset(
    {
        "Mul_A",
        "Mul_H",
        "Mul_HT",
        "Mul_Dinv",
        "Mul_Ainv",
    }
)
PASS_OPS = frozenset({"Pass_N", "Pass_M", "Pass_S"})
ADD_OPS = frozenset({"Add_N", "Add_M", "Add_S"})
SUB_OPS = frozenset({"Sub_N", "Sub_M", "Sub_S"})


def _scalar(value: torch.Tensor, block_idx: int | None = None) -> torch.Tensor:
    if torch.is_tensor(value) and value.dim() > 1 and block_idx is not None:
        return value[block_idx]
    return value


def _float_tuple(value: torch.Tensor) -> Tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())


def expression_cost(expr: Expression) -> int:
    if isinstance(expr, (Zero, Constant, Variable)):
        return 0
    if isinstance(expr, LinearCombination):
        return max(0, len(expr.terms) - 1) + sum(
            expression_cost(term) for _, term in expr.terms
        )
    return 1 + sum(expression_cost(arg) for arg in expr.args)


def walk_expression(expr: Expression) -> Iterable[Expression]:
    yield expr
    if isinstance(expr, LinearCombination):
        for _, term in expr.terms:
            yield from walk_expression(term)
    elif isinstance(expr, Call):
        for arg in expr.args:
            yield from walk_expression(arg)


def variable_names(expr: Expression) -> set[str]:
    return {item.name for item in walk_expression(expr) if isinstance(item, Variable)}


def operation_counts(expressions: Iterable[Expression]) -> Counter:
    counts = Counter()
    for expr in expressions:
        for item in walk_expression(expr):
            if isinstance(item, Call):
                counts[item.name] += 1
            elif isinstance(item, LinearCombination) and len(item.terms) > 1:
                counts["linear_add"] += len(item.terms) - 1
    return counts


def unique_call_counts(expressions: Iterable[Expression]) -> Counter:
    calls = {
        item
        for expression in expressions
        for item in walk_expression(expression)
        if isinstance(item, Call)
    }
    return Counter(item.name for item in calls)


def expression_to_data(expr: Expression) -> dict:
    base = {"dtype": expr.dtype.name}
    if isinstance(expr, Zero):
        return {"kind": "zero", **base}
    if isinstance(expr, Constant):
        return {"kind": "constant", **base, "value": expr.value}
    if isinstance(expr, Variable):
        return {"kind": "variable", **base, "name": expr.name}
    if isinstance(expr, LinearCombination):
        return {
            "kind": "linear_combination",
            **base,
            "terms": [
                {"coefficient": coefficient, "expression": expression_to_data(term)}
                for coefficient, term in expr.terms
            ],
        }
    return {
        "kind": "call",
        **base,
        "name": expr.name,
        "registry": expr.registry,
        "output_index": expr.output_index,
        "params": list(expr.params),
        "args": [expression_to_data(arg) for arg in expr.args],
    }


class ExpressionSimplifier:
    """Canonicalises typed expressions while recording applied exact rules."""

    def __init__(self, zero_tolerance: float = 1e-12, distribute_linear: bool = False):
        self.zero_tolerance = float(zero_tolerance)
        self.distribute_linear = bool(distribute_linear)
        self.rule_counts = Counter()

    def zero(self, dtype: DataType) -> Zero:
        return Zero(dtype)

    def constant(self, value: float) -> Expression:
        value = float(value)
        if abs(value) <= self.zero_tolerance:
            return Zero(DataType.SCALAR)
        return Constant(DataType.SCALAR, value)

    def linear(
        self, dtype: DataType, terms: Iterable[Tuple[float, Expression]]
    ) -> Expression:
        accumulated: MutableMapping[Expression, float] = {}
        constants = 0.0
        for coefficient, expr in terms:
            coefficient = float(coefficient)
            if abs(coefficient) <= self.zero_tolerance or isinstance(expr, Zero):
                self.rule_counts["zero_term_removed"] += 1
                continue
            if expr.dtype != dtype:
                raise TypeError(
                    f"Cannot add {expr.dtype.name} expression to {dtype.name}"
                )
            if isinstance(expr, LinearCombination):
                self.rule_counts["nested_linear_sum_flattened"] += 1
                for inner_coefficient, inner_expr in expr.terms:
                    accumulated[inner_expr] = (
                        accumulated.get(inner_expr, 0.0)
                        + coefficient * inner_coefficient
                    )
            elif isinstance(expr, Constant):
                constants += coefficient * expr.value
            else:
                accumulated[expr] = accumulated.get(expr, 0.0) + coefficient

        if dtype == DataType.SCALAR and abs(constants) > self.zero_tolerance:
            const_expr = Constant(DataType.SCALAR, 1.0)
            accumulated[const_expr] = accumulated.get(const_expr, 0.0) + constants

        cleaned: List[Tuple[float, Expression]] = []
        for expr, coefficient in accumulated.items():
            if abs(coefficient) <= self.zero_tolerance:
                self.rule_counts["cancelled_like_terms"] += 1
                continue
            cleaned.append((coefficient, expr))
        cleaned.sort(key=lambda item: repr(item[1]))

        if not cleaned:
            return Zero(dtype)
        if len(cleaned) == 1 and abs(cleaned[0][0] - 1.0) <= self.zero_tolerance:
            self.rule_counts["unit_scale_removed"] += 1
            return cleaned[0][1]
        if len(cleaned) < len(accumulated):
            self.rule_counts["identical_terms_combined"] += len(accumulated) - len(
                cleaned
            )
        return LinearCombination(dtype, tuple(cleaned))

    def scale(self, coefficient: float, expr: Expression) -> Expression:
        return self.linear(expr.dtype, ((coefficient, expr),))

    def add(self, left: Expression, right: Expression) -> Expression:
        return self.linear(left.dtype, ((1.0, left), (1.0, right)))

    def subtract(self, left: Expression, right: Expression) -> Expression:
        return self.linear(left.dtype, ((1.0, left), (-1.0, right)))

    def call(
        self,
        dtype: DataType,
        name: str,
        args: Sequence[Expression],
        params: Sequence[float] = (),
        registry: str = "ordinary",
        output_index: int = 0,
    ) -> Expression:
        args = tuple(args)
        params = tuple(float(value) for value in params)

        if registry == "ordinary" and name in PASS_OPS:
            self.rule_counts["identity_pass_removed"] += 1
            return args[0]
        if registry == "ordinary" and name in ADD_OPS:
            if args[0] == args[1]:
                self.rule_counts["duplicate_add_collapsed"] += 1
            return self.add(args[0], args[1])
        if registry == "ordinary" and name in SUB_OPS:
            if args[0] == args[1]:
                self.rule_counts["self_subtraction_removed"] += 1
            return self.subtract(args[0], args[1])
        if registry == "ordinary" and name == "Mix_N" and len(args) == 2:
            if args[0] == args[1]:
                self.rule_counts["duplicate_mix_collapsed"] += 1
                return args[0]
            alpha = 1.0 / (1.0 + math.exp(-params[0]))
            self.rule_counts["mix_expanded_to_affine_sum"] += 1
            return self.linear(dtype, ((alpha, args[0]), (1.0 - alpha, args[1])))
        if registry == "ordinary" and name in LINEAR_UNARY_OPS:
            if isinstance(args[0], Zero):
                self.rule_counts["linear_map_of_zero_removed"] += 1
                return Zero(dtype)
            if isinstance(args[0], LinearCombination) and len(args[0].terms) == 1:
                coefficient, term = args[0].terms[0]
                self.rule_counts["scalar_pulled_through_linear_map"] += 1
                return self.scale(
                    coefficient,
                    self.call(dtype, name, (term,), params, registry, output_index),
                )
            if self.distribute_linear and isinstance(args[0], LinearCombination):
                self.rule_counts["linear_map_distributed"] += 1
                return self.linear(
                    dtype,
                    (
                        (
                            coefficient,
                            self.call(
                                dtype, name, (term,), params, registry, output_index
                            ),
                        )
                        for coefficient, term in args[0].terms
                    ),
                )
        if registry == "ordinary" and name == "Cut_N":
            if isinstance(args[0], Zero):
                self.rule_counts["clip_of_zero_removed"] += 1
                return Zero(dtype)
            if isinstance(args[0], Call):
                inner = args[0]
                if inner.registry == "ordinary" and inner.name == "Cut_N":
                    outer_limit = abs(params[0])
                    inner_limit = abs(inner.params[0])
                    self.rule_counts["nested_clip_merged"] += 1
                    return Call(
                        dtype, name, inner.args, (min(outer_limit, inner_limit), 0.0)
                    )
        if (
            registry == "ordinary"
            and name == "SoftSlicer"
            and isinstance(args[0], Zero)
        ):
            # The 4-PAM support is symmetric around zero, hence its posterior
            # mean at a zero observation is exactly zero for every variance.
            self.rule_counts["symmetric_denoiser_at_zero_removed"] += 1
            return Zero(dtype)
        if registry == "ordinary" and name in {
            "Norm2_M",
            "Norm2_M_Scaled",
            "Mean_Abs_M",
        }:
            if isinstance(args[0], Zero):
                self.rule_counts["reduction_of_zero_removed"] += 1
                return Zero(dtype)
        if registry == "ordinary" and name == "Div_S" and isinstance(args[0], Zero):
            self.rule_counts["zero_dividend_removed"] += 1
            return Zero(dtype)
        return Call(dtype, name, args, params, registry, int(output_index))


@dataclass
class LayerExtraction:
    outputs: Dict[DataType, Expression]
    independent: List[Tuple[DataType, Expression]]
    active_operator_terms: int


@dataclass
class IterationFormula:
    index: int
    phi_n: Expression
    phi_m: Expression
    psi_n: Expression
    psi_m: Expression
    phi_states: List[Tuple[str, DataType, Expression]] = field(default_factory=list)
    psi_states: List[Tuple[str, DataType, Expression]] = field(default_factory=list)


@dataclass
class FormulaProgram:
    iterations: List[IterationFormula]
    simplifier_rules: Counter
    active_operator_terms: int
    threshold: float
    selected_operator_counts: Counter = field(default_factory=Counter)

    def roots(self) -> List[Expression]:
        roots: List[Expression] = []
        for item in self.iterations:
            roots.extend((item.phi_n, item.phi_m, item.psi_n, item.psi_m))
            roots.extend(expr for _, _, expr in item.phi_states)
            roots.extend(expr for _, _, expr in item.psi_states)
        return roots

    def live_named_expressions(self) -> List[Tuple[str, Expression]]:
        definitions: Dict[str, Expression] = {}
        mandatory: List[str] = []
        for item in self.iterations:
            definitions[f"p{item.index}"] = item.phi_n
            definitions[f"pm{item.index}"] = item.phi_m
            definitions[f"q{item.index}"] = item.psi_n
            definitions[f"qm{item.index}"] = item.psi_m
            definitions.update((name, expr) for name, _, expr in item.phi_states)
            definitions.update((name, expr) for name, _, expr in item.psi_states)
            mandatory.extend((f"p{item.index}", f"q{item.index}"))

        live = set(mandatory)
        queue = list(mandatory)
        while queue:
            name = queue.pop()
            expression = definitions[name]
            for dependency in variable_names(expression):
                if dependency in definitions and dependency not in live:
                    live.add(dependency)
                    queue.append(dependency)
        return [
            (name, expression)
            for name, expression in definitions.items()
            if name in live
        ]

    def to_data(self) -> dict:
        return {
            "threshold": self.threshold,
            "active_operator_terms_before_reachability_and_inlining": self.active_operator_terms,
            "selected_connected_operator_counts": dict(
                sorted(self.selected_operator_counts.items())
            ),
            "simplifier_rule_counts": dict(sorted(self.simplifier_rules.items())),
            "iterations": [
                {
                    "index": item.index,
                    "phi_n": expression_to_data(item.phi_n),
                    "phi_m": expression_to_data(item.phi_m),
                    "psi_n": expression_to_data(item.psi_n),
                    "psi_m": expression_to_data(item.psi_m),
                    "phi_states": [
                        {
                            "name": name,
                            "dtype": dtype.name,
                            "expression": expression_to_data(expr),
                        }
                        for name, dtype, expr in item.phi_states
                    ],
                    "psi_states": [
                        {
                            "name": name,
                            "dtype": dtype.name,
                            "expression": expression_to_data(expr),
                        }
                        for name, dtype, expr in item.psi_states
                    ],
                }
                for item in self.iterations
            ],
        }


class TypedFormulaExtractor:
    def __init__(self, model: DASDetector, threshold: float):
        self.model = model
        self.cfg = model.cfg
        self.threshold = float(threshold)
        # Distributing a matrix operator through a sum is algebraically valid
        # but can turn one matrix-vector product into many.  The readable form
        # therefore preserves the factored, minimum-MatVec representation.
        self.simplifier = ExpressionSimplifier(distribute_linear=False)

    def _source_expression(
        self,
        node,
        edge_key: str,
        required_type: DataType,
        slots: Sequence[Mapping[DataType, Expression]],
    ) -> Expression | None:
        active = [index for index, slot in enumerate(slots) if required_type in slot]
        if not active:
            return None
        weights = _scalar(node.edge_weights[edge_key]).detach().float()
        source = max(active, key=lambda i: float(weights[i]))
        return slots[source][required_type]

    def _ordinary_outputs(
        self,
        node,
        slots: Sequence[Mapping[DataType, Expression]],
    ) -> Tuple[Dict[DataType, Expression], int]:
        outputs: Dict[DataType, Expression] = {}
        active_terms = 0
        for out_name, op_indices in node.valid_ops_map.items():
            output_type = DataType[out_name]
            weights = _scalar(node.op_weights[out_name]).detach().float()
            terms: List[Tuple[float, Expression]] = []
            for local_index, op_index in enumerate(op_indices):
                coefficient = float(weights[local_index].item())
                if abs(coefficient) < self.threshold:
                    continue
                op = node.ops_registry[op_index]
                args: List[Expression] = []
                valid = True
                for arg_index, required_type in enumerate(op["in"]):
                    operand = self._source_expression(
                        node,
                        f"op{op_index}_arg{arg_index}",
                        required_type,
                        slots,
                    )
                    if operand is None:
                        valid = False
                        break
                    args.append(operand)
                if not valid:
                    continue
                params: Tuple[float, ...] = ()
                if op["learnable"]:
                    params = _float_tuple(
                        node.internal_params[node.param_map[op_index]]
                    )
                expression = self.simplifier.call(
                    output_type, op["name"], args, params=params
                )
                terms.append((coefficient, expression))
                active_terms += 1
            if terms:
                outputs[output_type] = self.simplifier.linear(output_type, terms)
        return outputs, active_terms

    def extract_layer(
        self,
        layer,
        input_slots: Sequence[Mapping[DataType, Expression]],
    ) -> LayerExtraction:
        slots: List[Dict[DataType, Expression]] = [dict(slot) for slot in input_slots]
        independent: List[Tuple[DataType, Expression]] = []
        active_terms = 0
        for node in layer.nodes:
            combined, count = self._ordinary_outputs(node, slots)
            slots.append(combined)
            active_terms += count

        outputs: Dict[DataType, Expression] = {}
        for output_type in layer.output_types:
            search_indices = list(reversed(layer.compute_output_indices))
            search_indices.extend(range(layer.start_node_idx - 1, -1, -1))
            selected = next(
                (
                    slots[index][output_type]
                    for index in search_indices
                    if output_type in slots[index]
                ),
                Zero(output_type),
            )
            outputs[output_type] = selected
        return LayerExtraction(outputs, independent, active_terms)

    @staticmethod
    def _slot(dtype: DataType, name: str) -> Dict[DataType, Expression]:
        return {dtype: Variable(dtype, name)}

    def extract(self) -> FormulaProgram:
        n_memory: List[str] = []
        m_memory: List[str] = []
        state_memory: Dict[DataType, List[str]] = {dtype: [] for dtype in DataType}
        iterations: List[IterationFormula] = []
        active_terms = 0

        for k, (phi, psi) in enumerate(zip(self.model.phis, self.model.psis)):
            phi_inputs = [
                self._slot(DataType.VEC_M, f"r{k}"),
                (
                    {DataType.VEC_M: Zero(DataType.VEC_M)}
                    if k == 0
                    else self._slot(DataType.VEC_M, f"r{k - 1}")
                ),
                (
                    {DataType.VEC_N: Zero(DataType.VEC_N)}
                    if k == 0
                    else self._slot(DataType.VEC_N, f"x{k}")
                ),
                {DataType.SCALAR: Constant(DataType.SCALAR, 1.0)},
            ]
            phi_inputs.extend(self._slot(DataType.VEC_N, name) for name in n_memory)
            phi_inputs.extend(self._slot(DataType.VEC_M, name) for name in m_memory)
            for dtype in DataType:
                phi_inputs.extend(
                    self._slot(dtype, name) for name in state_memory[dtype]
                )

            phi_result = self.extract_layer(phi, phi_inputs)
            p_name, pm_name = f"p{k}", f"pm{k}"
            n_memory.append(p_name)
            m_memory.append(pm_name)
            phi_state_defs: List[Tuple[str, DataType, Expression]] = []
            for output_type, expression in phi_result.independent:
                name = f"phis{k}_{output_type.name.lower()}_{len(state_memory[output_type])}"
                state_memory[output_type].append(name)
                phi_state_defs.append((name, output_type, expression))

            psi_inputs = [
                (
                    {DataType.VEC_N: Zero(DataType.VEC_N)}
                    if k == 0
                    else self._slot(DataType.VEC_N, f"d{k}")
                ),
                (
                    {DataType.VEC_N: Zero(DataType.VEC_N)}
                    if k <= 1
                    else self._slot(DataType.VEC_N, f"d{k - 1}")
                ),
                {DataType.SCALAR: Constant(DataType.SCALAR, 1.0)},
            ]
            psi_inputs.extend(self._slot(DataType.VEC_N, name) for name in n_memory)
            psi_inputs.extend(self._slot(DataType.VEC_M, name) for name in m_memory)
            for dtype in DataType:
                psi_inputs.extend(
                    self._slot(dtype, name) for name in state_memory[dtype]
                )

            psi_result = self.extract_layer(psi, psi_inputs)
            q_name, qm_name = f"q{k}", f"qm{k}"
            n_memory.append(q_name)
            m_memory.append(qm_name)
            psi_state_defs: List[Tuple[str, DataType, Expression]] = []
            for output_type, expression in psi_result.independent:
                name = f"psis{k}_{output_type.name.lower()}_{len(state_memory[output_type])}"
                state_memory[output_type].append(name)
                psi_state_defs.append((name, output_type, expression))

            iterations.append(
                IterationFormula(
                    index=k,
                    phi_n=phi_result.outputs[DataType.VEC_N],
                    phi_m=phi_result.outputs[DataType.VEC_M],
                    psi_n=psi_result.outputs[DataType.VEC_N],
                    psi_m=psi_result.outputs[DataType.VEC_M],
                    phi_states=phi_state_defs,
                    psi_states=psi_state_defs,
                )
            )
            active_terms += (
                phi_result.active_operator_terms + psi_result.active_operator_terms
            )

        return FormulaProgram(
            iterations=iterations,
            simplifier_rules=self.simplifier.rule_counts.copy(),
            active_operator_terms=active_terms,
            threshold=self.threshold,
        )
