#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

struct Pair { float r; float i; };

__device__ __forceinline__ float pam4_value(unsigned int symbol) {
    constexpr float a = 0.31622776601683794f;
    return symbol == 0u ? -a : (symbol == 1u ? -3.0f * a
        : (symbol == 2u ? a : 3.0f * a));
}

__device__ __forceinline__ void warp_bitonic_sort(
    float& key, unsigned int& value) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float other_key = __shfl_xor_sync(0xffffffffu, key, stride);
            const unsigned int other_value =
                __shfl_xor_sync(0xffffffffu, value, stride);
            const bool ascending = (lane & size) == 0;
            const bool lower_lane = (lane & stride) == 0;
            const bool take_other = (lower_lane == ascending)
                ? (other_key < key) : (other_key > key);
            if (take_other) { key = other_key; value = other_value; }
        }
    }
}



__global__ void grouped_kbest_preprocess_8(
    const float* __restrict__ h_input, float* __restrict__ lower_output,
    float* __restrict__ r_pam_output, int channels) {
    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= channels) return;
    const int tid = static_cast<int>(threadIdx.x);
    const Pair* h = reinterpret_cast<const Pair*>(h_input);
    __shared__ Pair channel_h[16 * 8];
    __shared__ Pair gram_complex[8 * 8];
    __shared__ float lower_real[16 * 16];
    __shared__ float inverse_pivot[16];

    if (tid < 128) channel_h[tid] = h[channel * 16 * 8 + tid];
    if (tid < 64) gram_complex[tid] = Pair{0.0f, 0.0f};
    for (int index = tid; index < 16 * 16; index += 128)
        lower_real[index] = 0.0f;
    __syncthreads();

    // All 64 compact complex Gram entries are independent.
    if (tid < 64) {
        const int row = tid >> 3;
        const int column = tid & 7;
        Pair value{0.0f, 0.0f};
        #pragma unroll
        for (int observation = 0; observation < 16; ++observation) {
            const Pair left = channel_h[observation * 8 + row];
            const Pair right = channel_h[observation * 8 + column];
            value.r = fmaf(left.r, right.r,
                fmaf(left.i, right.i, value.r));
            value.i = fmaf(left.r, right.i,
                fmaf(-left.i, right.r, value.i));
        }
        gram_complex[tid] = value;
    }
    __syncthreads();

    // Exact real 16x16 Cholesky.  Independent rows within each column run in
    // parallel while barriers retain the original factorization dependencies.
    #pragma unroll
    for (int pivot = 0; pivot < 16; ++pivot) {
        if (tid == pivot) {
            const int stream = pivot & 7;
            const Pair entry = gram_complex[stream * 8 + stream];
            const float gram_value = entry.r;
            float sum = 0.0f;
            #pragma unroll
            for (int previous = 0; previous < pivot; ++previous) {
                const float own = lower_real[pivot * 16 + previous];
                sum = fmaf(own, own, sum);
            }
            const float diagonal = sqrtf(fmaxf(gram_value - sum, 1.0e-12f));
            lower_real[pivot * 16 + pivot] = diagonal;
            inverse_pivot[pivot] = 1.0f / diagonal;
        }
        __syncthreads();
        if (tid < 16 && tid > pivot) {
            const int row = tid;
            const int row_stream = row & 7;
            const int column_stream = pivot & 7;
            const Pair entry = gram_complex[row_stream * 8 + column_stream];
            const float gram_value = row < 8
                ? (pivot < 8 ? entry.r : -entry.i)
                : (pivot < 8 ? entry.i : entry.r);
            float sum = 0.0f;
            #pragma unroll
            for (int previous = 0; previous < pivot; ++previous)
                sum = fmaf(lower_real[row * 16 + previous],
                    lower_real[pivot * 16 + previous], sum);
            lower_real[row * 16 + pivot] =
                (gram_value - sum) * inverse_pivot[pivot];
        }
        __syncthreads();
    }

    for (int index = tid; index < 136; index += 128) {
        int row = 0;
        int offset = index;
        while (offset > row) { offset -= row + 1; ++row; }
        const int column = offset;
        lower_output[channel * 136 + index] = lower_real[row * 16 + column];
    }
    __syncthreads();
    for (int item = tid; item < 136 * 4; item += 128) {
        const int upper = item >> 2;
        const int symbol = item & 3;
        int row = 0;
        int offset = upper;
        while (offset >= 16 - row) {
            offset -= 16 - row;
            ++row;
        }
        const int column = row + offset;
        const int lower_index = column * (column + 1) / 2 + row;
        r_pam_output[channel * 136 * 4 + item] =
            lower_output[channel * 136 + lower_index]
            * pam4_value(static_cast<unsigned int>(symbol));
    }
}

__global__ __launch_bounds__(512) void grouped_kbest_8_kernel(
    const float* __restrict__ h_input,
    const float* __restrict__ lower_input,
    const float* __restrict__ r_pam_input,
    const float* __restrict__ y, float* __restrict__ output,
    int vectors_per_channel, int persistent_blocks) {
    constexpr int K = 8;
    constexpr int SAMPLE_THREADS = 32;
    constexpr int SAMPLES_PER_BLOCK = 16;
    const int channel = static_cast<int>(blockIdx.y);
    const int sample_group = threadIdx.x / SAMPLE_THREADS;
    const int thread = threadIdx.x - sample_group * SAMPLE_THREADS;
    const Pair* h_group = reinterpret_cast<const Pair*>(h_input)
        + channel * 16 * 8;
    lower_input += channel * 136;
    r_pam_input += channel * 136 * 4;
    y += channel * vectors_per_channel * 32;
    output += channel * vectors_per_channel * 16;
    __shared__ Pair h_shared[16 * 8];
    __shared__ float lower_shared[136];
    __shared__ float r_pam_shared[136 * 4];
    __shared__ float z_storage[SAMPLES_PER_BLOCK * 16];
    __shared__ float survivor_metric_storage[SAMPLES_PER_BLOCK * K];
    __shared__ unsigned int survivor_code_storage[SAMPLES_PER_BLOCK * K];
    __shared__ float residual_center_storage[SAMPLES_PER_BLOCK * K];
    float* z_shared = z_storage + sample_group * 16;
    float* survivor_metric = survivor_metric_storage + sample_group * K;
    unsigned int* survivor_code = survivor_code_storage + sample_group * K;
    float* residual_center = residual_center_storage + sample_group * K;
    float* r_pam = r_pam_shared;

    for (int index = threadIdx.x; index < 16 * 8; index += 512)
        h_shared[index] = h_group[index];
    for (int index = threadIdx.x; index < 136; index += 512)
        lower_shared[index] = lower_input[index];
    for (int index = threadIdx.x; index < 136 * 4; index += 512)
        r_pam_shared[index] = r_pam_input[index];
    __syncthreads();

    for (int tile = blockIdx.x * SAMPLES_PER_BLOCK;
         tile < vectors_per_channel;
         tile += persistent_blocks * SAMPLES_PER_BLOCK) {
        const int batch = tile + sample_group;
        const bool active = batch < vectors_per_channel;
        if (thread < 16) {
            const int coordinate = thread;
            const int stream = coordinate & 7;
            float matched = 0.0f;
            #pragma unroll
            for (int observation = 0; observation < 16; ++observation) {
                const Pair h_value = h_shared[observation * 8 + stream];
                const float yr = active ? y[batch * 32 + observation] : 0.0f;
                const float yi = active ? y[batch * 32 + observation + 16] : 0.0f;
                matched = coordinate < 8
                    ? fmaf(h_value.r, yr, fmaf(h_value.i, yi, matched))
                    : fmaf(-h_value.i, yr, fmaf(h_value.r, yi, matched));
            }
            float transformed = 0.0f;
            #pragma unroll
            for (int row = 0; row < 16; ++row) {
                float value = matched;
                #pragma unroll
                for (int column = 0; column < row; ++column) {
                    const float coefficient = lower_shared[
                        row * (row + 1) / 2 + column];
                    value = fmaf(-coefficient,
                        __shfl_sync(0x0000ffffu, transformed, column, 16), value);
                }
                if (coordinate == row) {
                    const float diagonal = lower_shared[
                        row * (row + 1) / 2 + row];
                    transformed = value / diagonal;
                }
            }
            z_shared[coordinate] = transformed;
        }
        if (thread < K) {
            survivor_metric[thread] = 3.402823466e+38F;
            survivor_code[thread] = 0xffffffffu;
        }
        if (thread == 0) { survivor_metric[0] = 0.0f; survivor_code[0] = 0u; }
        __syncthreads();

        {
        // Tree depth 0: coordinate 15; exact global K-best.
        if (thread < 1) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            // No already-decided suffix at this depth.
            residual_center[thread] = z_shared[15] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 4) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[540 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 30);
        }
        if (thread < 4) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 1: coordinate 14; exact global K-best.
        if (thread < 4) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[536 + symbol];
            }
            residual_center[thread] = z_shared[14] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 16) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[532 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 28);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 2: coordinate 13; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[524 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[528 + symbol];
            }
            residual_center[thread] = z_shared[13] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[520 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 26);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 3: coordinate 12; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[508 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[512 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[516 + symbol];
            }
            residual_center[thread] = z_shared[12] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[504 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 24);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 4: coordinate 11; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[488 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[492 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[496 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[500 + symbol];
            }
            residual_center[thread] = z_shared[11] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[484 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 22);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 5: coordinate 10; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[464 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[468 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[472 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[476 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[480 + symbol];
            }
            residual_center[thread] = z_shared[10] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[460 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 20);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 6: coordinate 9; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[436 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[440 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[444 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[448 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[452 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[456 + symbol];
            }
            residual_center[thread] = z_shared[9] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[432 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 18);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 7: coordinate 8; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[404 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[408 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[412 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[416 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[420 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[424 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[428 + symbol];
            }
            residual_center[thread] = z_shared[8] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[400 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 16);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 8: coordinate 7; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[368 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[372 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[376 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[380 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[384 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[388 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[392 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[396 + symbol];
            }
            residual_center[thread] = z_shared[7] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[364 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 14);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 9: coordinate 6; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[328 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[332 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[336 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[340 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[344 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[348 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[352 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[356 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[360 + symbol];
            }
            residual_center[thread] = z_shared[6] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[324 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 12);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 10: coordinate 5; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[284 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[288 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[292 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[296 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[300 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[304 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[308 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[312 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[316 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[320 + symbol];
            }
            residual_center[thread] = z_shared[5] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[280 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 10);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 11: coordinate 4; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 10) & 3u;
                suffix += r_pam[236 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[240 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[244 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[248 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[252 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[256 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[260 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[264 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[268 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[272 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[276 + symbol];
            }
            residual_center[thread] = z_shared[4] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[232 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 8);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 12: coordinate 3; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 8) & 3u;
                suffix += r_pam[184 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 10) & 3u;
                suffix += r_pam[188 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[192 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[196 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[200 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[204 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[208 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[212 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[216 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[220 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[224 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[228 + symbol];
            }
            residual_center[thread] = z_shared[3] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[180 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 6);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 13: coordinate 2; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 6) & 3u;
                suffix += r_pam[128 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 8) & 3u;
                suffix += r_pam[132 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 10) & 3u;
                suffix += r_pam[136 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[140 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[144 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[148 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[152 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[156 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[160 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[164 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[168 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[172 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[176 + symbol];
            }
            residual_center[thread] = z_shared[2] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[124 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 4);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 14: coordinate 1; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 4) & 3u;
                suffix += r_pam[68 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 6) & 3u;
                suffix += r_pam[72 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 8) & 3u;
                suffix += r_pam[76 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 10) & 3u;
                suffix += r_pam[80 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[84 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[88 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[92 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[96 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[100 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[104 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[108 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[112 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[116 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[120 + symbol];
            }
            residual_center[thread] = z_shared[1] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[64 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 2);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        {
        // Tree depth 15: coordinate 0; exact global K-best.
        if (thread < 8) {
            const unsigned int parent_code = survivor_code[thread];
            float suffix = 0.0f;
            {
                const unsigned int symbol = (parent_code >> 2) & 3u;
                suffix += r_pam[4 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 4) & 3u;
                suffix += r_pam[8 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 6) & 3u;
                suffix += r_pam[12 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 8) & 3u;
                suffix += r_pam[16 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 10) & 3u;
                suffix += r_pam[20 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 12) & 3u;
                suffix += r_pam[24 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 14) & 3u;
                suffix += r_pam[28 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 16) & 3u;
                suffix += r_pam[32 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 18) & 3u;
                suffix += r_pam[36 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 20) & 3u;
                suffix += r_pam[40 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 22) & 3u;
                suffix += r_pam[44 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 24) & 3u;
                suffix += r_pam[48 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 26) & 3u;
                suffix += r_pam[52 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 28) & 3u;
                suffix += r_pam[56 + symbol];
            }
            {
                const unsigned int symbol = (parent_code >> 30) & 3u;
                suffix += r_pam[60 + symbol];
            }
            residual_center[thread] = z_shared[0] - suffix;
        }
        __syncthreads();
        float candidate_metric = 3.402823466e+38F;
        unsigned int candidate_code = 0xffffffffu;
        if (thread < 32) {
            const int parent = thread >> 2;
            const int symbol = thread & 3;
            const float residual = residual_center[parent]
                - r_pam[0 + symbol];
            candidate_metric = survivor_metric[parent] + residual * residual;
            candidate_code = survivor_code[parent]
                | (static_cast<unsigned int>(symbol) << 0);
        }
        warp_bitonic_sort(candidate_metric, candidate_code);
        __syncthreads();
        if (thread < 8) {
            survivor_metric[thread] = candidate_metric;
            survivor_code[thread] = candidate_code;
        }
        __syncthreads();
        }

        if (active && thread < 16) {
            const unsigned int symbol =
                (survivor_code[0] >> (2 * thread)) & 3u;
            output[batch * 16 + thread] = pam4_value(symbol);
        }
        __syncthreads();
    }
}

template<bool Preprocess>
int launch_grouped_kbest_8_impl(
    const float* h, const float* y, float* output,
    float* lower, float* r_pam, int channels, int vectors_per_channel,
    int blocks, cudaStream_t stream) {
    if (!h || !y || !output || !lower || !r_pam || channels <= 0 ||
        vectors_per_channel <= 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if constexpr (Preprocess) {
        grouped_kbest_preprocess_8<<<channels, 128, 0, stream>>>(
            h, lower, r_pam, channels);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    grouped_kbest_8_kernel<<<dim3(blocks, channels), 512, 0, stream>>>(
        h, lower, r_pam, y, output, vectors_per_channel, blocks);
    return static_cast<int>(cudaGetLastError());
}

#define GROUPED_KBEST_SIGNATURE(NAME) extern "C" int NAME(     int task, const float* h, const float* y, const float* noise, float* output,     float* w0, float* w1, float* w2, float* w3, float* w4, float* w5,     int channels, int vectors_per_channel, int blocks, cudaStream_t stream)

GROUPED_KBEST_SIGNATURE(launch_grouped_kbest_8_full) {
    (void)task; (void)noise; (void)w2; (void)w3; (void)w4; (void)w5;
    return launch_grouped_kbest_8_impl<true>(h, y, output, w0, w1,
        channels, vectors_per_channel, blocks, stream);
}
GROUPED_KBEST_SIGNATURE(launch_grouped_kbest_8_core) {
    (void)task; (void)noise; (void)w2; (void)w3; (void)w4; (void)w5;
    return launch_grouped_kbest_8_impl<false>(h, y, output, w0, w1,
        channels, vectors_per_channel, blocks, stream);
}
