from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from ..model import DataType
from .formula import (
    Call,
    Constant,
    Expression,
    LinearCombination,
    Variable,
    Zero,
    expression_cost,
    unique_call_counts,
    variable_names,
    walk_expression,
)

HIGH_COMPLEXITY = {"Mul_A", "Mul_H", "Mul_HT", "Mul_Ainv"}


def float_literal(value: float) -> str:
    if value == 0.0:
        return "0.0f"
    text = f"{float(value):.17g}"
    # CUDA parses ``1f`` as a user-defined literal; integral values still
    # need a decimal point (``1.0f``).
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


FIXED_COMPLEX_COMMON_SOURCE = r"""#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

struct Pair { float r; float i; };
struct Quad { Pair lo; Pair hi; };
struct Oct { Pair v0; Pair v1; Pair v2; Pair v3; };
__device__ __forceinline__ Pair pzero() { return Pair{0.0f, 0.0f}; }
__device__ __forceinline__ Quad qzero() { return Quad{pzero(), pzero()}; }
__device__ __forceinline__ Oct ozero() {
    return Oct{pzero(), pzero(), pzero(), pzero()};
}
__device__ __forceinline__ float softplus_stable(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}
__device__ __forceinline__ float pam4_denoise_invvar(
    float value, float inverse_variance) {
    constexpr float amplitude = 0.31622776601683794f;
    const float u = 2.0f * amplitude * fabsf(value) * inverse_variance;
    const float delta = 2.0f * u - 0.8f * inverse_variance;
    const float adjacent = expf(-fabsf(delta));
    const float opposite = expf(-2.0f * u);
    const float opposite3 = opposite * opposite * opposite;
    const float w3 = delta >= 0.0f ? 1.0f : adjacent;
    const float w1 = delta >= 0.0f ? adjacent : 1.0f;
    const float wm1 = w1 * opposite;
    const float wm3 = w3 * opposite3;
    const float numerator = 3.0f * w3 + w1 - wm1 - 3.0f * wm3;
    const float denominator = w3 + w1 + wm1 + wm3;
    return copysignf(amplitude * numerator / denominator, value);
}
__device__ __forceinline__ float pam4_denoise_tanh(
    float value, float inverse_variance, float outer_weight) {
    // Algebraically identical 4-PAM posterior mean.  Factoring the four
    // likelihoods through z=tanh(2*a*x/v) replaces two exponentials with one
    // tanh while retaining a stable positive denominator for the trained
    // inverse-variance range.
    constexpr float amplitude = 0.31622776601683794f;
    const float z = tanhf(2.0f * amplitude * value * inverse_variance);
    const float z2 = z * z;
    const float shared = fmaf(3.0f * outer_weight - 1.0f, z2, 0.0f);
    const float numerator_factor = shared + 1.0f + 9.0f * outer_weight;
    const float denominator = shared + 1.0f + outer_weight;
    return amplitude * z * numerator_factor / denominator;
}

"""


def _constant_scalar(expression: Expression) -> float | None:
    if isinstance(expression, Constant):
        return float(expression.value)
    if isinstance(expression, Zero):
        return 0.0
    return None


def _softslicer_inverse_variance(expression: Call) -> float | None:
    scalar = _constant_scalar(expression.args[1])
    if scalar is None:
        return None
    argument = torch.tensor(
        expression.params[0] * scalar + expression.params[1],
        dtype=torch.float32,
    )
    variance = F.softplus(argument) + torch.tensor(1.0e-4, dtype=torch.float32)
    return float(torch.reciprocal(variance).item())


def _softslicer_outer_weight(inverse_variance: float) -> float:
    """Return exp(-8*a^2/v)=exp(-0.8/v) in float32."""
    value = torch.tensor(-0.8 * inverse_variance, dtype=torch.float32)
    return float(torch.exp(value).item())


class KernelEmitter:
    """Emit one pruned detector using four lanes per receive vector."""

    _SUPPORTED = {
        "Mul_A",
        "Mul_Ainv",
        "Mul_Dinv",
        "Mul_HT",
        "Cut_N",
        "SoftSlicer",
        "Norm2_M",
        "Div_S",
    }

    def __init__(
        self,
        program,
        task_index,
        rate,
        warps_per_block,
        slicer_mode,
        vectorized_input,
        assume_aligned,
    ):
        self.program = program
        self.task_index = int(task_index)
        self.rate = float(rate)
        self.warps_per_block = int(warps_per_block)
        self.lines: List[str] = []
        self.cache: Dict[Expression, str] = {}
        self.temp_index = 0
        self.live_names = {name for name, _ in program.live_named_expressions()}
        roots = [expression for _, expression in program.live_named_expressions()]
        self.variables = set().union(*(variable_names(root) for root in roots))
        self.reference_counts = Counter(
            item for root in roots for item in walk_expression(root)
        )
        self.expected_calls = unique_call_counts(roots)
        self.emitted_calls = Counter()
        self.effective_residuals = []
        self.residual_sources = {}
        self.m_state_expressions = {}
        self.slicer_mode = slicer_mode
        self.vectorized_input = vectorized_input
        self.assume_aligned = assume_aligned
        self.mc, self.nc = 16, 8
        self.half_nc, self.samples_per_warp, self.observations_per_lane = 4, 8, 4
        self.matched_name = "matched_raw"
        self.fused_residual_gram_actions = 0

    def new_temp(self) -> str:
        name = f"v{self.temp_index}"
        self.temp_index += 1
        return name

    @staticmethod
    def scalar_sum(terms: Iterable[Tuple[float, str]]) -> str:
        pieces = []
        for index, (coefficient, code) in enumerate(terms):
            magnitude = abs(coefficient)
            body = (
                code
                if abs(magnitude - 1.0) < 1e-15
                else f"{float_literal(magnitude)} * ({code})"
            )
            if index == 0:
                pieces.append(("-" if coefficient < 0 else "") + body)
            else:
                pieces.append((" - " if coefficient < 0 else " + ") + body)
        return "".join(pieces) if pieces else "0.0f"

    def emit_scalar_sum(self, expression, terms):
        code = self.scalar_sum(terms)
        if self.reference_counts[expression] >= 2 and expression_cost(expression) >= 2:
            name = self.new_temp()
            self.lines.append(f"    float {name} = active ? ({code}) : 0.0f;")
            self.cache[expression] = name
            return name
        return f"({code})"

    def emit_residual(self, iteration: int) -> None:
        name = f"r{iteration}"
        if name not in self.variables:
            return
        self.effective_residuals.append(name)
        if iteration != 0:
            self.residual_sources[name] = f"x{iteration}"
            self.lines.append(
                f"    // EFFECTIVE_RESIDUAL {name}: materialization fused into H^H(y-Hx)"
            )
            return
        # Mul_HT(r0) is replaced directly by the already fused matched filter.
        self.lines.append(
            "    // EFFECTIVE_RESIDUAL r0: consumed by fused complex H^H y"
        )

    def _denoiser_call(
        self,
        value: str,
        inverse_variance: float | None,
        inverse_code: str,
    ) -> str:
        if (
            self.slicer_mode == "tanh4"
            and inverse_variance is not None
            and inverse_variance <= 4
        ):
            outer = float_literal(_softslicer_outer_weight(inverse_variance))
            return f"pam4_denoise_tanh({value}, {inverse_code}, {outer})"
        return f"pam4_denoise_invvar({value}, {inverse_code})"

    @staticmethod
    def _quad_component(code: str, half: str, component: str) -> str:
        return f"{code}.{half}.{component}"

    def emit_linear(self, expression: LinearCombination) -> str:
        terms = [
            (coefficient, self.emit_expression(child))
            for coefficient, child in expression.terms
        ]
        if expression.dtype == DataType.VEC_N:
            name = self.new_temp()
            values = {}
            for half in ("lo", "hi"):
                for component in ("r", "i"):
                    values[(half, component)] = self.scalar_sum(
                        (
                            coefficient,
                            self._quad_component(code, half, component),
                        )
                        for coefficient, code in terms
                    )
            self.lines.extend(
                [
                    f"    Quad {name};",
                    f"    {name}.lo.r = active ? ({values[('lo', 'r')]}) : 0.0f;",
                    f"    {name}.lo.i = active ? ({values[('lo', 'i')]}) : 0.0f;",
                    f"    {name}.hi.r = active ? ({values[('hi', 'r')]}) : 0.0f;",
                    f"    {name}.hi.i = active ? ({values[('hi', 'i')]}) : 0.0f;",
                ]
            )
            self.cache[expression] = name
            return name
        if expression.dtype == DataType.VEC_M:
            raise NotImplementedError("live M-state is kept symbolically")
        return self.emit_scalar_sum(expression, terms)

    def emit_expression(self, expression: Expression) -> str:
        if expression in self.cache:
            return self.cache[expression]
        if isinstance(expression, Zero):
            return "qzero()" if expression.dtype == DataType.VEC_N else "0.0f"
        if isinstance(expression, Constant):
            if expression.dtype == DataType.VEC_N:
                value = float_literal(expression.value)
                return f"Quad{{Pair{{{value}, {value}}}, Pair{{{value}, {value}}}}}"
            return float_literal(expression.value)
        if isinstance(expression, Variable):
            return expression.name
        if isinstance(expression, LinearCombination):
            return self.emit_linear(expression)
        if isinstance(expression, Call):
            return self.emit_call(expression)
        raise TypeError(type(expression))

    def named_assignment(self, name: str, expression: Expression) -> None:
        if expression.dtype == DataType.VEC_M:
            self.m_state_expressions[name] = expression
            self.lines.append(
                f"    // M-state {name} retained symbolically; no materialization"
            )
            return
        code = self.emit_expression(expression)
        if expression.dtype == DataType.VEC_N:
            self.lines.append(f"    Quad {name} = {code};")
        elif expression.dtype == DataType.SCALAR:
            self.lines.append(f"    float {name} = {code};")
        else:
            raise NotImplementedError(expression.dtype)

    def emit_iteration(self, item) -> None:
        k = item.index
        self.lines.append(f"    // ITERATION {k}")
        self.emit_residual(k)
        if f"d{k}" in self.variables:
            if k == 0:
                self.lines.append(f"    Quad d{k} = qzero();")
            else:
                self.lines.extend(
                    [
                        f"    Quad d{k};",
                        f"    d{k}.lo.r = active ? x{k}.lo.r - x{k - 1}.lo.r : 0.0f;",
                        f"    d{k}.lo.i = active ? x{k}.lo.i - x{k - 1}.lo.i : 0.0f;",
                        f"    d{k}.hi.r = active ? x{k}.hi.r - x{k - 1}.hi.r : 0.0f;",
                        f"    d{k}.hi.i = active ? x{k}.hi.i - x{k - 1}.hi.i : 0.0f;",
                    ]
                )
        self.named_assignment(f"p{k}", item.phi_n)
        if f"pm{k}" in self.live_names:
            self.named_assignment(f"pm{k}", item.phi_m)
        for state_name, _dtype, expression in item.phi_states:
            if state_name in self.live_names:
                self.named_assignment(state_name, expression)
        self.named_assignment(f"q{k}", item.psi_n)
        if f"qm{k}" in self.live_names:
            self.named_assignment(f"qm{k}", item.psi_m)
        for state_name, _dtype, expression in item.psi_states:
            if state_name in self.live_names:
                self.named_assignment(state_name, expression)
        previous = "x0" if k == 0 else f"x{k}"
        values = {}
        for half in ("lo", "hi"):
            for component in ("r", "i"):
                value = (
                    f"{previous}.{half}.{component} + "
                    f"p{k}.{half}.{component} + q{k}.{half}.{component}"
                )
                value = f"fminf(fmaxf({value}, -10.0f), 10.0f)"
                values[(half, component)] = value
        self.lines.extend(
            [
                f"    Quad x{k + 1};",
                f"    x{k + 1}.lo.r = active ? {values[('lo', 'r')]} : 0.0f;",
                f"    x{k + 1}.lo.i = active ? {values[('lo', 'i')]} : 0.0f;",
                f"    x{k + 1}.hi.r = active ? {values[('hi', 'r')]} : 0.0f;",
                f"    x{k + 1}.hi.i = active ? {values[('hi', 'i')]} : 0.0f;",
            ]
        )

    def _matrix_coefficient_half(
        self, matrix: str, half: str, column: str = "column"
    ) -> str:
        return f"{matrix}_{half}_local[{column}]"

    def emit_fixed_matvec(self, argument: str, name: str, matrix: str) -> None:
        coefficient_lo = self._matrix_coefficient_half(matrix, "lo")
        coefficient_hi = self._matrix_coefficient_half(matrix, "hi")
        self.lines.extend(
            [
                f"    Quad {name} = qzero();",
                "    if (active) {",
                "        #pragma unroll",
                "        for (int column = 0; column < NC; ++column) {",
                "            Pair source;",
                f"            if (column < {self.half_nc}) {{",
                f"                source.r = __shfl_sync(mask, {argument}.lo.r, column, {self.half_nc});",
                f"                source.i = __shfl_sync(mask, {argument}.lo.i, column, {self.half_nc});",
                "            } else {",
                f"                source.r = __shfl_sync(mask, {argument}.hi.r, column - {self.half_nc}, {self.half_nc});",
                f"                source.i = __shfl_sync(mask, {argument}.hi.i, column - {self.half_nc}, {self.half_nc});",
                "            }",
                f"            const Pair coefficient_lo = {coefficient_lo};",
                f"            const Pair coefficient_hi = {coefficient_hi};",
                f"            {name}.lo.r = fmaf(coefficient_lo.r, source.r,",
                f"                fmaf(-coefficient_lo.i, source.i, {name}.lo.r));",
                f"            {name}.lo.i = fmaf(coefficient_lo.r, source.i,",
                f"                fmaf(coefficient_lo.i, source.r, {name}.lo.i));",
                f"            {name}.hi.r = fmaf(coefficient_hi.r, source.r,",
                f"                fmaf(-coefficient_hi.i, source.i, {name}.hi.r));",
                f"            {name}.hi.i = fmaf(coefficient_hi.r, source.i,",
                f"                fmaf(coefficient_hi.i, source.r, {name}.hi.i));",
                "        }",
                "    }",
            ]
        )

    def emit_gram_action(self, argument: str, name: str) -> None:
        a_name = f"{name}_a"
        self.emit_fixed_matvec(argument, a_name, "A")
        self.lines.extend(
            [
                f"    Quad {name};",
                f"    {name}.lo.r = active ? fmaf(-sigma2, {argument}.lo.r, {a_name}.lo.r) : 0.0f;",
                f"    {name}.lo.i = active ? fmaf(-sigma2, {argument}.lo.i, {a_name}.lo.i) : 0.0f;",
                f"    {name}.hi.r = active ? fmaf(-sigma2, {argument}.hi.r, {a_name}.hi.r) : 0.0f;",
                f"    {name}.hi.i = active ? fmaf(-sigma2, {argument}.hi.i, {a_name}.hi.i) : 0.0f;",
            ]
        )
        self.fused_residual_gram_actions += 1

    def emit_ht_expression(self, expression: Expression, requested_name: str) -> str:
        if isinstance(expression, Zero):
            self.lines.append(f"    Quad {requested_name} = qzero();")
            return requested_name
        if isinstance(expression, Variable):
            source_name = expression.name
            if source_name in self.m_state_expressions:
                return self.emit_ht_expression(
                    self.m_state_expressions[source_name],
                    requested_name,
                )
            if source_name == "r0":
                self.lines.append(
                    f"    Quad {requested_name} = active ? matched_raw : qzero();"
                )
                return requested_name
            if source_name in self.residual_sources:
                gram_name = f"{requested_name}_gram_prediction"
                self.emit_gram_action(self.residual_sources[source_name], gram_name)
                self.lines.extend(
                    [
                        f"    Quad {requested_name};",
                        f"    {requested_name}.lo.r = active ? matched_raw.lo.r - {gram_name}.lo.r : 0.0f;",
                        f"    {requested_name}.lo.i = active ? matched_raw.lo.i - {gram_name}.lo.i : 0.0f;",
                        f"    {requested_name}.hi.r = active ? matched_raw.hi.r - {gram_name}.hi.r : 0.0f;",
                        f"    {requested_name}.hi.i = active ? matched_raw.hi.i - {gram_name}.hi.i : 0.0f;",
                    ]
                )
                return requested_name
            raise NotImplementedError(f"unknown fixed-H M-variable {source_name}")
        if isinstance(expression, LinearCombination):
            terms = []
            for index, (coefficient, child) in enumerate(expression.terms):
                child_name = self.emit_ht_expression(
                    child,
                    f"{requested_name}_term{index}",
                )
                terms.append((coefficient, child_name))
            values = {}
            for half in ("lo", "hi"):
                for component in ("r", "i"):
                    values[(half, component)] = self.scalar_sum(
                        (
                            coefficient,
                            f"{code}.{half}.{component}",
                        )
                        for coefficient, code in terms
                    )
            self.lines.extend(
                [
                    f"    Quad {requested_name};",
                    f"    {requested_name}.lo.r = active ? ({values[('lo', 'r')]}) : 0.0f;",
                    f"    {requested_name}.lo.i = active ? ({values[('lo', 'i')]}) : 0.0f;",
                    f"    {requested_name}.hi.r = active ? ({values[('hi', 'r')]}) : 0.0f;",
                    f"    {requested_name}.hi.i = active ? ({values[('hi', 'i')]}) : 0.0f;",
                ]
            )
            return requested_name
        raise NotImplementedError(
            f"cannot fuse fixed-H Mul_HT over {type(expression).__name__}"
        )

    def emit_call(self, expression: Call) -> str:
        if expression in self.cache:
            return self.cache[expression]
        op = expression.name
        if op not in self._SUPPORTED:
            raise NotImplementedError(op)
        self.emitted_calls[op] += 1
        name = self.new_temp()
        argument_expression = expression.args[0]
        if op == "Mul_HT":
            self.lines.append(
                "    // CSE_CALL Mul_HT (4-lane fixed-H Gram-domain fusion)"
            )
            self.emit_ht_expression(argument_expression, name)
        elif op in {"Mul_A", "Mul_Ainv"}:
            argument = self.emit_expression(argument_expression)
            matrix = "A" if op == "Mul_A" else "Ainv"
            self.lines.append(f"    // CSE_CALL {op} (4-lane fixed complex action)")
            self.emit_fixed_matvec(argument, name, matrix)
        elif op == "Mul_Dinv":
            argument = self.emit_expression(argument_expression)
            self.lines.extend(
                [
                    "    // CSE_CALL Mul_Dinv",
                    f"    Quad {name} = active ? Quad{{",
                    f"        Pair{{d_inv_lo * {argument}.lo.r, d_inv_lo * {argument}.lo.i}},",
                    f"        Pair{{d_inv_hi * {argument}.hi.r, d_inv_hi * {argument}.hi.i}}",
                    "    } : qzero();",
                ]
            )
        elif op == "Cut_N":
            argument = self.emit_expression(argument_expression)
            bound = float_literal(abs(expression.params[0]))

            def clip(half: str, component: str) -> str:
                value = f"{argument}.{half}.{component}"
                return f"fminf(fmaxf({value}, -{bound}), {bound})"

            self.lines.extend(
                [
                    "    // CSE_CALL Cut_N",
                    f"    Quad {name} = active ? Quad{{",
                    f"        Pair{{{clip('lo', 'r')}, {clip('lo', 'i')}}},",
                    f"        Pair{{{clip('hi', 'r')}, {clip('hi', 'i')}}}",
                    "    } : qzero();",
                ]
            )
        elif op == "SoftSlicer":
            argument = self.emit_expression(argument_expression)
            inverse_variance = _softslicer_inverse_variance(expression)
            self.lines.append("    // CSE_CALL SoftSlicer")
            if inverse_variance is None:
                scalar = self.emit_expression(expression.args[1])
                p0, p1 = expression.params[:2]
                self.lines.extend(
                    [
                        f"    const float {name}_variance = softplus_stable(",
                        f"        {float_literal(p0)} * ({scalar}) + {float_literal(p1)}) + 1.0e-4f;",
                        f"    const float {name}_inverse_variance = 1.0f / {name}_variance;",
                    ]
                )
                inverse_code = f"{name}_inverse_variance"
            else:
                inverse_code = float_literal(inverse_variance)
            denoise_lo_r = self._denoiser_call(
                f"{argument}.lo.r",
                inverse_variance,
                inverse_code,
            )
            denoise_lo_i = self._denoiser_call(
                f"{argument}.lo.i",
                inverse_variance,
                inverse_code,
            )
            denoise_hi_r = self._denoiser_call(
                f"{argument}.hi.r",
                inverse_variance,
                inverse_code,
            )
            denoise_hi_i = self._denoiser_call(
                f"{argument}.hi.i",
                inverse_variance,
                inverse_code,
            )
            self.lines.extend(
                [
                    f"    Quad {name} = active ? Quad{{",
                    f"        Pair{{{denoise_lo_r},",
                    f"             {denoise_lo_i}}},",
                    f"        Pair{{{denoise_hi_r},",
                    f"             {denoise_hi_i}}}",
                    "    } : qzero();",
                ]
            )
        elif op == "Norm2_M":
            if not (
                isinstance(argument_expression, Variable)
                and argument_expression.name == "r0"
            ):
                raise NotImplementedError("fixed complex Norm2_M requires r0")
            self.lines.extend(
                [
                    "    // CSE_CALL Norm2_M",
                    f"    float {name} = active ? r0_norm2 : 0.0f;",
                ]
            )
        elif op == "Div_S":
            numerator = self.emit_expression(expression.args[0])
            denominator = self.emit_expression(expression.args[1])
            self.lines.extend(
                [
                    "    // CSE_CALL Div_S",
                    f"    float {name} = active ? ({numerator})",
                    f"        / (fabsf({denominator}) + 1.0e-6f) : 0.0f;",
                ]
            )
        self.cache[expression] = name
        return name

    def emit(self) -> Tuple[str, dict]:
        self.lines = ["    Quad x0 = qzero();"]
        for item in self.program.iterations:
            self.emit_iteration(item)
        if self.emitted_calls != self.expected_calls:
            raise RuntimeError(
                f"task {self.task_index}: CSE emission mismatch "
                f"expected={self.expected_calls}, emitted={self.emitted_calls}"
            )
        body = "\n".join(self.lines)
        iterations = len(self.program.iterations)
        matrix_registers: List[str] = []
        matrix_source = "A_shared"
        inverse_source = "Ainv_shared"
        matrix_registers.extend(
            [
                "    Pair A_lo_local[NC], A_hi_local[NC];",
                "    Pair Ainv_lo_local[NC], Ainv_hi_local[NC];",
                "    #pragma unroll",
                "    for (int j = 0; j < NC; ++j) {",
                f"        A_lo_local[j] = {matrix_source}[coordinate * NC + j];",
                f"        A_hi_local[j] = {matrix_source}[(coordinate + {self.half_nc}) * NC + j];",
                f"        Ainv_lo_local[j] = {inverse_source}[coordinate * NC + j];",
                f"        Ainv_hi_local[j] = {inverse_source}[(coordinate + {self.half_nc}) * NC + j];",
                "    }",
            ]
        )
        matrix_register_source = "\n".join(matrix_registers)
        matrix_storage_source = r"""    __shared__ Pair H_shared[MC * NC];
__shared__ Pair A_shared[NC * NC];
__shared__ Pair Ainv_shared[NC * NC];
for (int index = threadIdx.x; index < MC * NC; index += blockDim.x)
    H_shared[index] = H_input[index];
for (int index = threadIdx.x; index < NC * NC; index += blockDim.x) {
    A_shared[index] = A_input[index];
    Ainv_shared[index] = Ainv_input[index];
}
__syncthreads();"""
        h_source = "H_shared"
        needs_r0_norm = bool(self.expected_calls["Norm2_M"])
        norm_source = ""
        if needs_r0_norm:
            norm_lines = ["    float r0_norm2_lane = 0.0f;"]
            for observation in range(self.observations_per_lane):
                norm_lines.append(
                    f"    r0_norm2_lane = fmaf(y{observation}.r, y{observation}.r, "
                    f"fmaf(y{observation}.i, y{observation}.i, r0_norm2_lane));"
                )
            offset = self.half_nc // 2
            while offset:
                norm_lines.append(
                    "    r0_norm2_lane += __shfl_down_sync(mask, "
                    f"r0_norm2_lane, {offset}, {self.half_nc});"
                )
                offset //= 2
            norm_lines.append(
                "    const float r0_norm2 = __shfl_sync(mask, "
                f"r0_norm2_lane, 0, {self.half_nc}) * "
                f"(1.0f / {float_literal(2.0 * self.mc)});"
            )
            norm_source = "\n".join(norm_lines)
        slots = self.warps_per_block * self.samples_per_warp
        threads = self.warps_per_block * 32
        if self.vectorized_input:
            y_lines = ["    const float* const y_base = y + batch * MR;"]
            if self.observations_per_lane % 4 == 0:
                vector_width = 4
                vector_type = "float4"
                zero_value = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)"
                fields = "xyzw"
            elif self.observations_per_lane % 2 == 0:
                vector_width = 2
                vector_type = "float2"
                zero_value = "make_float2(0.0f, 0.0f)"
                fields = "xy"
            else:
                raise ValueError("vectorized pack2 input requires an even Nr/(Nt/2)")
            vectors_per_lane = self.observations_per_lane // vector_width
            for vector in range(vectors_per_lane):
                vector_index = f"coordinate * {vectors_per_lane} + {vector}"
                for component, base in (("real", "y_base"), ("imag", "y_base + MC")):
                    y_lines.extend(
                        [
                            f"    const {vector_type} y_{component}{vector} = active",
                            f"        ? reinterpret_cast<const {vector_type}*>({base})[{vector_index}]",
                            f"        : {zero_value};",
                        ]
                    )
            for observation in range(self.observations_per_lane):
                vector = observation // vector_width
                field = fields[observation % vector_width]
                y_lines.append(
                    f"    Pair y{observation} = Pair{{y_real{vector}.{field}, "
                    f"y_imag{vector}.{field}}};"
                )
            y_load_source = "\n".join(y_lines)
        else:
            y_lines = []
            for observation in range(self.observations_per_lane):
                offset = f"coordinate * {self.observations_per_lane} + {observation}"
                y_lines.append(
                    f"    Pair y{observation} = active ? Pair{{"
                    f"y[batch * MR + {offset}], y[batch * MR + MC + {offset}]"
                    "} : pzero();"
                )
            y_load_source = "\n".join(y_lines)
        matched_lines = ["    if (active) {"]
        for observation in range(self.observations_per_lane):
            for source_lane in range(self.half_nc):
                receiver = observation + self.observations_per_lane * source_lane
                matched_lines.extend(
                    [
                        f"        const Pair source_{observation}_{source_lane} = Pair{{",
                        f"            __shfl_sync(mask, y{observation}.r, {source_lane}, {self.half_nc}),",
                        f"            __shfl_sync(mask, y{observation}.i, {source_lane}, {self.half_nc})}};",
                        f"        const Pair h_lo_{observation}_{source_lane} = "
                        f"{h_source}[{receiver} * NC + coordinate];",
                        f"        const Pair h_hi_{observation}_{source_lane} = "
                        f"{h_source}[{receiver} * NC + coordinate + {self.half_nc}];",
                        f"        matched_raw.lo.r = fmaf(h_lo_{observation}_{source_lane}.r, "
                        f"source_{observation}_{source_lane}.r, fmaf(h_lo_{observation}_{source_lane}.i, "
                        f"source_{observation}_{source_lane}.i, matched_raw.lo.r));",
                        f"        matched_raw.lo.i = fmaf(h_lo_{observation}_{source_lane}.r, "
                        f"source_{observation}_{source_lane}.i, fmaf(-h_lo_{observation}_{source_lane}.i, "
                        f"source_{observation}_{source_lane}.r, matched_raw.lo.i));",
                        f"        matched_raw.hi.r = fmaf(h_hi_{observation}_{source_lane}.r, "
                        f"source_{observation}_{source_lane}.r, fmaf(h_hi_{observation}_{source_lane}.i, "
                        f"source_{observation}_{source_lane}.i, matched_raw.hi.r));",
                        f"        matched_raw.hi.i = fmaf(h_hi_{observation}_{source_lane}.r, "
                        f"source_{observation}_{source_lane}.i, fmaf(-h_hi_{observation}_{source_lane}.i, "
                        f"source_{observation}_{source_lane}.r, matched_raw.hi.i));",
                    ]
                )
        matched_lines.append("    }")
        matched_source = "\n".join(matched_lines)
        output_store_source = f"""        output[batch * NR + coordinate] = x{iterations}.lo.r;
    output[batch * NR + coordinate + {self.half_nc}] = x{iterations}.hi.r;
    output[batch * NR + coordinate + {self.nc}] = x{iterations}.lo.i;
    output[batch * NR + coordinate + {self.nc + self.half_nc}] = x{iterations}.hi.i;"""
        if self.assume_aligned:
            active_source = r"""    constexpr bool active = true;
    constexpr unsigned mask = 0xffffffffu;"""
        else:
            active_source = r"""    const bool active = batch < batch_size;
    const unsigned mask = __ballot_sync(0xffffffffu, active);"""
        source = f"""// BEGIN_TASK {self.task_index:02d} RATE {self.rate:.9f}
__global__ __launch_bounds__({threads}) void dag_fixed_complex_pack2_task_{self.task_index:02d}(
    const Pair* __restrict__ H_input, const float* __restrict__ y,
    const Pair* __restrict__ A_input, const float* __restrict__ D_inv_input,
    const Pair* __restrict__ Ainv_input, const float* __restrict__ sigma2_input,
    float* __restrict__ output, int batch_size, int persistent_blocks) {{
    constexpr int NC = {self.nc}, MC = {self.mc}, NR = {2 * self.nc}, MR = {2 * self.mc}, SLOTS = {slots};
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane / {self.half_nc};
    const int coordinate = lane & {self.half_nc - 1};
    const int sample_slot = warp * {self.samples_per_warp} + group;
{matrix_storage_source}
{matrix_register_source}
    const float d_inv_lo = D_inv_input[coordinate];
    const float d_inv_hi = D_inv_input[coordinate + {self.half_nc}];
    const float sigma2 = sigma2_input[0];
    for (int tile = blockIdx.x * SLOTS; tile < batch_size;
         tile += persistent_blocks * SLOTS) {{
    const int batch = tile + sample_slot;
{active_source}
{y_load_source}
    Quad matched_raw = qzero();
{matched_source}
{norm_source}
{body}
    if (active) {{
{output_store_source}
    }}
    }}
}}
// END_TASK {self.task_index:02d}
"""
        return source


def emit_kernel(
    program,
    task_index,
    rate,
    warps_per_block,
    slicer_mode,
    vectorized_input,
    assume_aligned,
):
    kernel = KernelEmitter(
        program,
        task_index,
        rate,
        warps_per_block,
        slicer_mode,
        vectorized_input,
        assume_aligned,
    ).emit()
    return FIXED_COMPLEX_COMMON_SOURCE + "\n" + kernel
