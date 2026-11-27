from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DenseOffsets:
    weight: int
    bias: int


class ParameterPacker:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add_dense(self, state: dict[str, torch.Tensor], prefix: str) -> DenseOffsets:
        weight = state[f"{prefix}.weight"].detach().float().cpu().contiguous()
        bias = state[f"{prefix}.bias"].detach().float().cpu().contiguous()
        weight_offset = len(self.values)
        self.values.extend(float(value) for value in weight.reshape(-1))
        bias_offset = len(self.values)
        self.values.extend(float(value) for value in bias.reshape(-1))
        return DenseOffsets(weight_offset, bias_offset)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def float_literal(value: float) -> str:
    return float(value).hex() + "f"


def format_values(values: list[float], width: int = 6) -> str:
    rows = []
    for index in range(0, len(values), width):
        rows.append(
            "    " + ", ".join(float_literal(v) for v in values[index : index + width])
        )
    return ",\n".join(rows)


def format_ints(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def generate(checkpoint: Path) -> str:
    state = load_state(checkpoint)
    packer = ParameterPacker()
    rx0: list[DenseOffsets] = []
    rx1: list[DenseOffsets] = []
    tx0: list[DenseOffsets] = []
    tx1: list[DenseOffsets] = []
    for iteration in range(10):
        rx0.append(packer.add_dense(state, f"receive_processors.{iteration}.layers.0"))
        rx1.append(packer.add_dense(state, f"receive_processors.{iteration}.layers.1"))
        if iteration < 9:
            tx0.append(
                packer.add_dense(state, f"transmit_processors.{iteration}.layers.0")
            )
            tx1.append(
                packer.add_dense(state, f"transmit_processors.{iteration}.layers.1")
            )
    final0 = packer.add_dense(state, "transmit_processors.9.layers.0")
    final1 = packer.add_dense(state, "transmit_processors.9.layers.1")
    final2 = packer.add_dense(state, "transmit_processors.9.layers.2")
    if len(packer.values) != 7484:
        raise RuntimeError(
            f"expected 7484 ChannelNet parameters, packed {len(packer.values)}"
        )

    def offsets(items: list[DenseOffsets], field: str) -> str:
        return format_ints([getattr(item, field) for item in items])

    return f"""// Generated from {checkpoint.name}; do not edit by hand.
#include <cuda_runtime.h>
#include <math.h>

namespace {{

constexpr int MC = 16;
constexpr int NC = 8;
constexpr int REAL_RX = 32;
constexpr int REAL_TX = 16;
constexpr int FEATURES = 10;
constexpr int FINAL_WIDTH = 50;
constexpr int CLASSES = 4;
constexpr int CHANNEL_FLOATS = 512;  // row-major and transposed compact H
constexpr int FINAL_BUFFER_FLOATS = REAL_TX * FINAL_WIDTH;
constexpr int WARP_BUFFER_FLOATS = 2 * FINAL_BUFFER_FLOATS;

struct __align__(8) Pair {{ float r; float i; }};

__device__ __constant__ float PARAMS[{len(packer.values)}] = {{
{format_values(packer.values)}
}};

__device__ __constant__ int RX_W0[10] = {{{offsets(rx0, "weight")}}};
__device__ __constant__ int RX_B0[10] = {{{offsets(rx0, "bias")}}};
__device__ __constant__ int RX_W1[10] = {{{offsets(rx1, "weight")}}};
__device__ __constant__ int RX_B1[10] = {{{offsets(rx1, "bias")}}};
__device__ __constant__ int TX_W0[9] = {{{offsets(tx0, "weight")}}};
__device__ __constant__ int TX_B0[9] = {{{offsets(tx0, "bias")}}};
__device__ __constant__ int TX_W1[9] = {{{offsets(tx1, "weight")}}};
__device__ __constant__ int TX_B1[9] = {{{offsets(tx1, "bias")}}};

__device__ __forceinline__ float relu(float value) {{
    return fmaxf(value, 0.0f);
}}

__device__ __forceinline__ void mlp10x10x10(
    float (&features)[FEATURES], int w0, int b0, int w1, int b1) {{
    float hidden[FEATURES];
    #pragma unroll
    for (int output = 0; output < FEATURES; ++output) {{
        float sum = PARAMS[b0 + output];
        #pragma unroll
        for (int input = 0; input < FEATURES; ++input)
            sum = fmaf(PARAMS[w0 + output * FEATURES + input], features[input], sum);
        hidden[output] = relu(sum);
    }}
    #pragma unroll
    for (int output = 0; output < FEATURES; ++output) {{
        float sum = PARAMS[b1 + output];
        #pragma unroll
        for (int input = 0; input < FEATURES; ++input)
            sum = fmaf(PARAMS[w1 + output * FEATURES + input], hidden[input], sum);
        features[output] = relu(sum);
    }}
}}

template<int Warps>
__global__ void grouped_channelnet_kernel(
    const Pair* __restrict__ h,
    const float* __restrict__ y,
    float* __restrict__ output,
    int vectors_per_channel,
    int blocks_per_channel) {{
    extern __shared__ float shared[];
    Pair* h_row = reinterpret_cast<Pair*>(shared);
    Pair* h_column = h_row + MC * NC;
    float* warp_buffers = shared + CHANNEL_FLOATS;

    const int channel = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const Pair* source_h = h + channel * MC * NC;
    for (int linear = tid; linear < MC * NC; linear += blockDim.x) {{
        const Pair value = source_h[linear];
        const int observation = linear / NC;
        const int coordinate = linear - observation * NC;
        h_row[linear] = value;
        h_column[coordinate * MC + observation] = value;
    }}
    __syncthreads();

    float* final0 = warp_buffers + warp * WARP_BUFFER_FLOATS;
    float* final1 = final0 + FINAL_BUFFER_FLOATS;
    const unsigned mask = 0xffffffffu;

    for (int vector = static_cast<int>(blockIdx.x) * Warps + warp;
         vector < vectors_per_channel;
         vector += blocks_per_channel * Warps) {{
        const float* vector_y = y + (channel * vectors_per_channel + vector) * REAL_RX;
        float features[FEATURES];
        #pragma unroll
        for (int feature = 0; feature < FEATURES; ++feature)
            features[feature] = vector_y[lane];

        #pragma unroll
        for (int iteration = 0; iteration < 10; ++iteration) {{
            mlp10x10x10(
                features, RX_W0[iteration], RX_B0[iteration],
                RX_W1[iteration], RX_B1[iteration]);

            // Exact H^T propagation in the paper's real-equivalent system.
            #pragma unroll
            for (int feature = 0; feature < FEATURES; ++feature) {{
                float real_sum = 0.0f;
                float imag_sum = 0.0f;
                #pragma unroll
                for (int observation = 0; observation < MC; ++observation) {{
                    const float real_value = __shfl_sync(mask, features[feature], observation);
                    const float imag_value = __shfl_sync(mask, features[feature], observation + MC);
                    if (lane < NC) {{
                        const Pair coefficient = h_column[lane * MC + observation];
                        real_sum = fmaf(coefficient.r, real_value,
                            fmaf(coefficient.i, imag_value, real_sum));
                        imag_sum = fmaf(-coefficient.i, real_value,
                            fmaf(coefficient.r, imag_value, imag_sum));
                    }}
                }}
                const int source = lane & (NC - 1);
                real_sum = __shfl_sync(mask, real_sum, source);
                imag_sum = __shfl_sync(mask, imag_sum, source);
                float value = lane < NC ? real_sum : imag_sum;
                if (lane < REAL_TX) {{
                    const int previous_index = feature * REAL_TX + lane;
                    if (iteration > 0) value += final0[previous_index];
                    final0[previous_index] = value;
                    features[feature] = value;
                }}
            }}

            if (iteration == 9) {{
                // 10 -> 50.  Feature-major layout makes each half warp read
                // one shared weight row and write adjacent coordinates.
                for (int linear = lane; linear < FINAL_WIDTH * REAL_TX; linear += 32) {{
                    const int neuron = linear / REAL_TX;
                    const int coordinate = linear - neuron * REAL_TX;
                    float sum = PARAMS[{final0.bias} + neuron];
                    #pragma unroll
                    for (int input = 0; input < FEATURES; ++input) {{
                        const float value = __shfl_sync(mask, features[input], coordinate);
                        sum = fmaf(PARAMS[{final0.weight} + neuron * FEATURES + input], value, sum);
                    }}
                    final0[linear] = relu(sum);
                }}
                __syncwarp(mask);

                // 50 -> 50.
                for (int linear = lane; linear < FINAL_WIDTH * REAL_TX; linear += 32) {{
                    const int neuron = linear / REAL_TX;
                    const int coordinate = linear - neuron * REAL_TX;
                    float sum = PARAMS[{final1.bias} + neuron];
                    #pragma unroll
                    for (int input = 0; input < FINAL_WIDTH; ++input)
                        sum = fmaf(
                            PARAMS[{final1.weight} + neuron * FINAL_WIDTH + input],
                            final0[input * REAL_TX + coordinate], sum);
                    final1[linear] = relu(sum);
                }}
                __syncwarp(mask);

                if (lane < REAL_TX) {{
                    float logits[CLASSES];
                    #pragma unroll
                    for (int klass = 0; klass < CLASSES; ++klass) {{
                        float sum = PARAMS[{final2.bias} + klass];
                        #pragma unroll
                        for (int input = 0; input < FINAL_WIDTH; ++input)
                            sum = fmaf(
                                PARAMS[{final2.weight} + klass * FINAL_WIDTH + input],
                                final1[input * REAL_TX + lane], sum);
                        logits[klass] = sum;
                    }}
                    float maximum = logits[0];
                    #pragma unroll
                    for (int klass = 1; klass < CLASSES; ++klass)
                        maximum = fmaxf(maximum, logits[klass]);
                    constexpr float levels[CLASSES] = {{
                        -0.9486832980505138f, -0.31622776601683794f,
                         0.31622776601683794f, 0.9486832980505138f}};
                    float denominator = 0.0f;
                    float numerator = 0.0f;
                    #pragma unroll
                    for (int klass = 0; klass < CLASSES; ++klass) {{
                        const float probability = expf(logits[klass] - maximum);
                        denominator += probability;
                        numerator = fmaf(probability, levels[klass], numerator);
                    }}
                    output[(channel * vectors_per_channel + vector) * REAL_TX + lane] =
                        numerator / denominator;
                }}
                __syncwarp(mask);
                continue;
            }}

            if (lane < REAL_TX)
                mlp10x10x10(
                    features, TX_W0[iteration], TX_B0[iteration],
                    TX_W1[iteration], TX_B1[iteration]);

            // Exact H propagation.  The column-major compact layout makes
            // all 16 active lanes read adjacent coefficients for each k.
            #pragma unroll
            for (int feature = 0; feature < FEATURES; ++feature) {{
                float real_sum = 0.0f;
                float imag_sum = 0.0f;
                #pragma unroll
                for (int coordinate = 0; coordinate < NC; ++coordinate) {{
                    const float real_value = __shfl_sync(mask, features[feature], coordinate);
                    const float imag_value = __shfl_sync(mask, features[feature], coordinate + NC);
                    if (lane < MC) {{
                        const Pair coefficient = h_column[coordinate * MC + lane];
                        real_sum = fmaf(coefficient.r, real_value,
                            fmaf(-coefficient.i, imag_value, real_sum));
                        imag_sum = fmaf(coefficient.i, real_value,
                            fmaf(coefficient.r, imag_value, imag_sum));
                    }}
                }}
                const int source = lane & (MC - 1);
                real_sum = __shfl_sync(mask, real_sum, source);
                imag_sum = __shfl_sync(mask, imag_sum, source);
                features[feature] = (lane < MC ? real_sum : imag_sum) - vector_y[lane];
            }}
        }}
    }}
}}

template<int Warps>
int launch_channelnet(
    const float* h, const float* y, float* output,
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream) {{
    if (!h || !y || !output || channels <= 0 || vectors_per_channel <= 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    constexpr int threads = 32 * Warps;
    constexpr int shared_bytes =
        (CHANNEL_FLOATS + Warps * WARP_BUFFER_FLOATS) * sizeof(float);
    if (shared_bytes > 48 * 1024) {{
        cudaError_t attribute = cudaFuncSetAttribute(
            grouped_channelnet_kernel<Warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        if (attribute != cudaSuccess) return static_cast<int>(attribute);
    }}
    grouped_channelnet_kernel<Warps>
        <<<dim3(blocks, channels), threads, shared_bytes, stream>>>(
            reinterpret_cast<const Pair*>(h), y, output,
            vectors_per_channel, blocks);
    return static_cast<int>(cudaGetLastError());
}}

}}  // namespace

#define GROUPED_SIGNATURE(NAME) \\
extern "C" int NAME( \\
    int task, const float* h, const float* y, const float* noise, float* output, \\
    float* w0, float* w1, float* w2, float* w3, float* w4, float* w5, \\
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream)

#define DEFINE_CHANNELNET(W) \\
GROUPED_SIGNATURE(launch_grouped_channelnet_w##W##_core) {{ \\
    (void)task; (void)noise; (void)w0; (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; \\
    return launch_channelnet<W>(h, y, output, channels, vectors_per_channel, blocks, stream); \\
}} \\
GROUPED_SIGNATURE(launch_grouped_channelnet_w##W##_full) {{ \\
    (void)task; (void)noise; (void)w0; (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; \\
    return launch_channelnet<W>(h, y, output, channels, vectors_per_channel, blocks, stream); \\
}}

DEFINE_CHANNELNET(1)
DEFINE_CHANNELNET(2)
DEFINE_CHANNELNET(3)
DEFINE_CHANNELNET(4)
DEFINE_CHANNELNET(5)
DEFINE_CHANNELNET(6)
DEFINE_CHANNELNET(7)
DEFINE_CHANNELNET(8)
"""
