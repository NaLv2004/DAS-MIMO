#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Exact complex-packed lowering for the repository's 16x8 complex MIMO
// represented externally as a 32x16 real-equivalent system.  AMP remains
// separable on the real and imaginary PAM axes, but its linear algebra has the
// exact 8x8 complex Hermitian structure.  EP can use that structure only for
// the initial alpha=2 posterior; after the first denoiser the real/imag site
// precisions are independent, so every later factorisation intentionally stays
// 16-dimensional and real.

namespace {

constexpr int NC = 8;
constexpr int NR = 16;
constexpr int MC = 16;
constexpr int MR = 32;
constexpr float PAM_A = 0.31622776601683793320f;
constexpr float EPS = 1.0e-12f;
constexpr float VAR_FLOOR = 5.0e-7f;

#ifndef EP_ITERATIONS
#define EP_ITERATIONS 7
#endif

struct Pair { float r; float i; };

__device__ __forceinline__ Pair pzero() { return Pair{0.0f, 0.0f}; }

__device__ __forceinline__ Pair pmul(Pair a, Pair b) {
    return Pair{
        fmaf(a.r, b.r, -a.i * b.i),
        fmaf(a.r, b.i,  a.i * b.r),
    };
}

__device__ __forceinline__ void pam4_posterior(
    float value, float variance, float* mean, float* posterior_variance) {
    const float inverse_variance = 1.0f / fmaxf(variance, EPS);
    const float u = 2.0f * PAM_A * fabsf(value) * inverse_variance;
    const float delta = 2.0f * u - 0.8f * inverse_variance;
    const float adjacent = __expf(-fabsf(delta));
    const float opposite = __expf(-2.0f * u);
    const float opposite3 = opposite * opposite * opposite;
    const float w3 = delta >= 0.0f ? 1.0f : adjacent;
    const float w1 = delta >= 0.0f ? adjacent : 1.0f;
    const float wm1 = w1 * opposite;
    const float wm3 = w3 * opposite3;
    const float inverse_sum = 1.0f / (w3 + w1 + wm1 + wm3);
    const float signed_first = PAM_A
        * (3.0f * w3 + w1 - wm1 - 3.0f * wm3) * inverse_sum;
    const float first = copysignf(signed_first, value);
    const float second = PAM_A * PAM_A
        * (9.0f * w3 + w1 + wm1 + 9.0f * wm3) * inverse_sum;
    *mean = first;
    *posterior_variance = fmaxf(second - first * first, 0.0f);
}

__device__ __forceinline__ void pam4_ep_moments(
    float cavity_mean, float cavity_variance,
    float& posterior_mean, float& posterior_variance) {
    const float inverse_two_variance = 0.5f / fmaxf(cavity_variance, EPS);
    const float d0 = cavity_mean + 3.0f * PAM_A;
    const float d1 = cavity_mean + PAM_A;
    const float d2 = cavity_mean - PAM_A;
    const float d3 = cavity_mean - 3.0f * PAM_A;
    const float l0 = -d0 * d0 * inverse_two_variance;
    const float l1 = -d1 * d1 * inverse_two_variance;
    const float l2 = -d2 * d2 * inverse_two_variance;
    const float l3 = -d3 * d3 * inverse_two_variance;
    const float maximum = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
    const float w0 = __expf(l0 - maximum);
    const float w1 = __expf(l1 - maximum);
    const float w2 = __expf(l2 - maximum);
    const float w3 = __expf(l3 - maximum);
    const float inverse_sum = 1.0f / (w0 + w1 + w2 + w3);
    posterior_mean = PAM_A * (-3.0f * w0 - w1 + w2 + 3.0f * w3)
        * inverse_sum;
    const float second = 0.1f * (9.0f * w0 + w1 + w2 + 9.0f * w3)
        * inverse_sum;
    posterior_variance = fmaxf(
        second - posterior_mean * posterior_mean, VAR_FLOOR);
}

// -------------------------------------------------------------------------
// AMP: every lane owns one complex stream, hence four samples per warp.
// -------------------------------------------------------------------------

template<int Threads, int EffectiveUpdates, bool FixedChannel,
         bool PackedPerH = false>
__global__ void complex_amp_kernel(
    const float* __restrict__ H_input,
    const float* __restrict__ gram_input,
    const float* __restrict__ y,
    const float* __restrict__ noise_var,
    float* __restrict__ output,
    int batch_size,
    int persistent_blocks) {

    const int grouped_channel = static_cast<int>(blockIdx.y);
    if constexpr (FixedChannel) {
        H_input += grouped_channel * MC * NC * 2;
        gram_input += grouped_channel * NC * NC * 2;
        y += grouped_channel * batch_size * MR;
        noise_var += grouped_channel;
        output += grouped_channel * batch_size * NR;
    }
    const float grouped_amp_y_scale = FixedChannel ? 0.25f : 1.0f;
    constexpr int SLOTS = Threads / NC;
    __shared__ float fixed_h[MC * NC * 2];
    __shared__ float fixed_g[NC * NC * 2];
    if constexpr (FixedChannel) {
        for (int index = threadIdx.x; index < MC * NC * 2; index += Threads)
            fixed_h[index] = H_input[index];
        for (int index = threadIdx.x; index < NC * NC * 2; index += Threads)
            fixed_g[index] = gram_input[index];
        __syncthreads();
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 3;
    const int coordinate = lane & 7;
    const int sample_slot = warp * 4 + group;
    Pair fixed_gram_row[NC];
    if constexpr (FixedChannel) {
        #pragma unroll
        for (int column = 0; column < NC; ++column) {
            const int index = (coordinate * NC + column) * 2;
            fixed_gram_row[column] = Pair{fixed_g[index], fixed_g[index + 1]};
        }
    }

    for (int tile = blockIdx.x * SLOTS; tile < batch_size;
         tile += persistent_blocks * SLOTS) {
        const int batch = tile + sample_slot;
        const bool active = batch < batch_size;
        // A complete 8-lane tile executes even for a tail slot.  Keeping the
        // tile itself in the shuffle mask avoids undefined shuffle behaviour
        // for inactive tail samples; their arithmetic is zeroed by `active`.
        const unsigned mask = 0xffu << (group * NC);
        Pair gram_row[NC];
        #pragma unroll
        for (int column = 0; column < NC; ++column) {
            if constexpr (FixedChannel) gram_row[column] = fixed_gram_row[column];
            else gram_row[column] = pzero();
        }
        Pair matched = pzero();

        #pragma unroll
        for (int observation = 0; observation < MC; ++observation) {
            float h_r, h_i;
            if constexpr (FixedChannel) {
                const int h_index = (observation * NC + coordinate) * 2;
                h_r = fixed_h[h_index];
                h_i = fixed_h[h_index + 1];
            } else if constexpr (PackedPerH) {
                const Pair* const h_complex =
                    reinterpret_cast<const Pair*>(H_input);
                const Pair h_value = active
                    ? h_complex[(batch * MC + observation) * NC + coordinate]
                    : pzero();
                h_r = h_value.r;
                h_i = h_value.i;
            } else {
                h_r = active
                    ? H_input[(batch * MR + observation) * NR + coordinate]
                    : 0.0f;
                h_i = active
                    ? H_input[(batch * MR + observation + MC) * NR + coordinate]
                    : 0.0f;
            }
            // Every tile lane names the same two addresses.  CUDA coalescing
            // handles each as a single broadcast transaction, avoiding the
            // extra shuffle instruction a manual lane-0 broadcast would add.
            float y_r, y_i;
            if constexpr (PackedPerH) {
                const Pair* const y_complex =
                    reinterpret_cast<const Pair*>(y);
                const Pair y_value = active
                    ? y_complex[batch * MC + observation] : pzero();
                y_r = y_value.r;
                y_i = y_value.i;
            } else {
                y_r = active ? (grouped_amp_y_scale * y[batch * MR + observation]) : 0.0f;
                y_i = active ? (grouped_amp_y_scale * y[batch * MR + observation + MC]) : 0.0f;
            }
            matched.r = fmaf(h_r, y_r, fmaf(h_i, y_i, matched.r));
            matched.i = fmaf(h_r, y_i, fmaf(-h_i, y_r, matched.i));
            if constexpr (!FixedChannel) {
                #pragma unroll
                for (int column = 0; column < NC; ++column) {
                    const float other_r = __shfl_sync(mask, h_r, column, NC);
                    const float other_i = __shfl_sync(mask, h_i, column, NC);
                    gram_row[column].r = fmaf(h_r, other_r,
                        fmaf(h_i, other_i, gram_row[column].r));
                    gram_row[column].i = fmaf(h_r, other_i,
                        fmaf(-h_i, other_r, gram_row[column].i));
                }
            }
        }
        if constexpr (!FixedChannel) {
            #pragma unroll
            for (int column = 0; column < NC; ++column) {
                gram_row[column].r *= 0.0625f;
                gram_row[column].i *= 0.0625f;
            }
            matched.r *= 0.0625f;
            matched.i *= 0.0625f;
        }
        const float n0 = active
            ? fmaxf(FixedChannel ? noise_var[0] : noise_var[batch] * 0.0625f, EPS)
            : 1.0f;
        Pair residual = matched;
        Pair estimate = pzero();
        float tau = 0.5f / n0;

        #pragma unroll
        for (int iteration = 0; iteration < EffectiveUpdates; ++iteration) {
            const Pair z{estimate.r + residual.r, estimate.i + residual.i};
            const float effective_variance = fmaxf(n0 * (1.0f + tau), EPS);
            Pair candidate;
            float variance_r = 0.0f, variance_i = 0.0f;
            if (active) {
                pam4_posterior(z.r, effective_variance, &candidate.r, &variance_r);
                pam4_posterior(z.i, effective_variance, &candidate.i, &variance_i);
            } else {
                candidate = pzero();
            }
            float variance_sum = active ? variance_r + variance_i : 0.0f;
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
                variance_sum += __shfl_down_sync(mask, variance_sum, offset, NC);
            const float tau_lane0 = coordinate == 0
                ? 0.5f * (variance_sum / 8.0f) / n0 : 0.0f;
            const float tau_candidate = __shfl_sync(mask, tau_lane0, 0, NC);
            const float onsager = tau_candidate / (1.0f + tau);
            Pair gram_candidate = pzero();
            #pragma unroll
            for (int column = 0; column < NC; ++column) {
                const Pair source{
                    __shfl_sync(mask, candidate.r, column, NC),
                    __shfl_sync(mask, candidate.i, column, NC),
                };
                gram_candidate.r = fmaf(gram_row[column].r, source.r,
                    fmaf(-gram_row[column].i, source.i, gram_candidate.r));
                gram_candidate.i = fmaf(gram_row[column].r, source.i,
                    fmaf( gram_row[column].i, source.r, gram_candidate.i));
            }
            const Pair next{
                matched.r - gram_candidate.r + onsager * residual.r,
                matched.i - gram_candidate.i + onsager * residual.i,
            };
            estimate.r = 0.5f * (estimate.r + candidate.r);
            estimate.i = 0.5f * (estimate.i + candidate.i);
            tau = 0.5f * (tau + tau_candidate);
            residual.r = 0.5f * (residual.r + next.r);
            residual.i = 0.5f * (residual.i + next.i);
        }
        if (active) {
            if constexpr (PackedPerH) {
                Pair* const output_complex = reinterpret_cast<Pair*>(output);
                output_complex[batch * NC + coordinate] = Pair{
                    estimate.r + residual.r, estimate.i + residual.i};
            } else {
                output[batch * NR + coordinate] = estimate.r + residual.r;
                output[batch * NR + coordinate + NC] = estimate.i + residual.i;
            }
        }
    }
}

// -------------------------------------------------------------------------
// EP helpers.  A 16-lane tile owns one sample.  Its first eight lanes build
// the physical complex Gram and solve the initial equal-site posterior.
// -------------------------------------------------------------------------

__device__ __forceinline__ float tile16_shuffle(float value, int source) {
    return __shfl_sync(0xffffffffu, value, source, 16);
}

__device__ __forceinline__ Pair tile16_shuffle_pair(Pair value, int source) {
    return Pair{
        __shfl_sync(0xffffffffu, value.r, source, 16),
        __shfl_sync(0xffffffffu, value.i, source, 16),
    };
}

__device__ __forceinline__ void initial_complex_ep_posterior(
    int coordinate, Pair matched, const Pair (&gram_row)[NC],
    Pair& mean, float& marginal_variance) {
    Pair lower[NC];
    #pragma unroll
    for (int j = 0; j < NC; ++j) lower[j] = pzero();
    float own_inverse_pivot = 0.0f;
    #pragma unroll
    for (int column = 0; column < NC; ++column) {
        Pair sum = pzero();
        #pragma unroll
        for (int previous = 0; previous < column; ++previous) {
            const Pair other = tile16_shuffle_pair(lower[previous], column);
            sum.r = fmaf(lower[previous].r, other.r,
                fmaf(lower[previous].i, other.i, sum.r));
            sum.i = fmaf(lower[previous].i, other.r,
                fmaf(-lower[previous].r, other.i, sum.i));
        }
        float inverse_pivot = 0.0f;
        if (coordinate == column) {
            lower[column].r = sqrtf(fmaxf(
                gram_row[column].r + 2.0f - sum.r, EPS));
            lower[column].i = 0.0f;
            inverse_pivot = 1.0f / lower[column].r;
            own_inverse_pivot = inverse_pivot;
        }
        inverse_pivot = tile16_shuffle(inverse_pivot, column);
        if (coordinate > column) {
            lower[column].r = (gram_row[column].r - sum.r) * inverse_pivot;
            lower[column].i = (gram_row[column].i - sum.i) * inverse_pivot;
        }
    }

    Pair forward = pzero();
    #pragma unroll
    for (int row = 0; row < NC; ++row) {
        Pair value = matched;
        #pragma unroll
        for (int column = 0; column < row; ++column) {
            const Pair fwd = tile16_shuffle_pair(forward, column);
            const Pair product = pmul(lower[column], fwd);
            value.r -= product.r;
            value.i -= product.i;
        }
        if (coordinate == row) {
            forward.r = value.r * own_inverse_pivot;
            forward.i = value.i * own_inverse_pivot;
        }
    }
    mean = pzero();
    #pragma unroll
    for (int row = NC - 1; row >= 0; --row) {
        Pair value = forward;
        #pragma unroll
        for (int source = row + 1; source < NC; ++source) {
            Pair coefficient = tile16_shuffle_pair(lower[row], source);
            coefficient.i = -coefficient.i;
            const Pair solution = tile16_shuffle_pair(mean, source);
            const Pair product = pmul(coefficient, solution);
            value.r -= product.r;
            value.i -= product.i;
        }
        if (coordinate == row) {
            mean.r = value.r * own_inverse_pivot;
            mean.i = value.i * own_inverse_pivot;
        }
    }

    Pair inverse_column[NC];
    #pragma unroll
    for (int j = 0; j < NC; ++j) inverse_column[j] = pzero();
    #pragma unroll
    for (int row = 0; row < NC; ++row) {
        const float inverse_diagonal = tile16_shuffle(
            own_inverse_pivot, row);
        Pair sum = pzero();
        #pragma unroll
        for (int inner = 0; inner < row; ++inner) {
            // Every lane named by the shuffle mask must execute the shuffle;
            // applying the triangular condition around the arithmetic (not
            // around the intrinsic) is required for defined warp behaviour.
            const Pair coefficient = tile16_shuffle_pair(lower[inner], row);
            if (inner >= coordinate) {
                const Pair product = pmul(coefficient, inverse_column[inner]);
                sum.r += product.r;
                sum.i += product.i;
            }
        }
        if (coordinate == row) {
            inverse_column[row] = Pair{inverse_diagonal, 0.0f};
        } else if (coordinate < row) {
            inverse_column[row] = Pair{
                -sum.r * inverse_diagonal, -sum.i * inverse_diagonal};
        }
    }
    marginal_variance = 0.0f;
    #pragma unroll
    for (int row = 0; row < NC; ++row) {
        if (row >= coordinate) {
            marginal_variance = fmaf(
                inverse_column[row].r, inverse_column[row].r,
                fmaf(inverse_column[row].i, inverse_column[row].i,
                     marginal_variance));
        }
    }
}

template<int Threads>
__global__ void per_h_hybrid_ep_kernel(
    const float* __restrict__ H,
    const float* __restrict__ y,
    const float* __restrict__ noise_var,
    float* __restrict__ output,
    int batch_size,
    int persistent_blocks) {
    constexpr int SLOTS = Threads / NR;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile_in_warp = lane >> 4;
    const int coordinate = lane & 15;
    const int sample_slot = warp * 2 + tile_in_warp;

    for (int tile = blockIdx.x * SLOTS; tile < batch_size;
         tile += persistent_blocks * SLOTS) {
        const int batch = tile + sample_slot;
        const bool active = batch < batch_size;
        const float sigma2 = active ? fmaxf(noise_var[batch], EPS) : 1.0f;
        const float inverse_noise = 1.0f / sigma2;
        Pair gram_row[NC];
        #pragma unroll
        for (int j = 0; j < NC; ++j) gram_row[j] = pzero();
        Pair matched_pair = pzero();
        #pragma unroll
        for (int observation = 0; observation < MC; ++observation) {
                const float h_r = active && coordinate < NC
                    ? H[(batch * MR + observation) * NR + coordinate] : 0.0f;
                const float h_i = active && coordinate < NC
                    ? H[(batch * MR + observation + MC) * NR + coordinate] : 0.0f;
                const float y_r = active && coordinate < NC
                    ? y[batch * MR + observation] : 0.0f;
                const float y_i = active && coordinate < NC
                    ? y[batch * MR + observation + MC] : 0.0f;
                matched_pair.r = fmaf(h_r, y_r,
                    fmaf(h_i, y_i, matched_pair.r));
                matched_pair.i = fmaf(h_r, y_i,
                    fmaf(-h_i, y_r, matched_pair.i));
                #pragma unroll
                for (int column = 0; column < NC; ++column) {
                    const float other_r = tile16_shuffle(h_r, column);
                    const float other_i = tile16_shuffle(h_i, column);
                    gram_row[column].r = fmaf(h_r, other_r,
                        fmaf(h_i, other_i, gram_row[column].r));
                    gram_row[column].i = fmaf(h_r, other_i,
                        fmaf(-h_i, other_r, gram_row[column].i));
                }
        }
        if (coordinate < NC) {
            matched_pair.r *= inverse_noise;
            matched_pair.i *= inverse_noise;
            #pragma unroll
            for (int column = 0; column < NC; ++column) {
                gram_row[column].r *= inverse_noise;
                gram_row[column].i *= inverse_noise;
            }
        }

        float gram_lower[NR];
        #pragma unroll
        for (int column = 0; column < NR; ++column) {
            const int row_stream = coordinate & 7;
            const int column_stream = column & 7;
            const float gr = tile16_shuffle(
                gram_row[column_stream].r, row_stream);
            const float gi = tile16_shuffle(
                gram_row[column_stream].i, row_stream);
            float value;
            if (coordinate < NC)
                value = column < NC ? gr : -gi;
            else
                value = column < NC ? gi : gr;
            gram_lower[column] = column <= coordinate ? value : 0.0f;
        }
        const float matched_i = tile16_shuffle(matched_pair.i, coordinate & 7);
        const float matched = coordinate < NC ? matched_pair.r : matched_i;

        Pair initial_mean_pair = pzero();
        float initial_variance = 0.0f;
        initial_complex_ep_posterior(
            coordinate, matched_pair, gram_row,
            initial_mean_pair, initial_variance);
        const float initial_mean_i = tile16_shuffle(
            initial_mean_pair.i, coordinate & 7);
        const float initial_variance_i = tile16_shuffle(
            initial_variance, coordinate & 7);
        float mean = coordinate < NC ? initial_mean_pair.r : initial_mean_i;
        float marginal_variance = coordinate < NC
            ? initial_variance : initial_variance_i;
#ifdef EP_DEBUG_INITIAL
        if (active) {
            output[batch * 32 + coordinate] = mean;
            output[batch * 32 + coordinate + 16] = marginal_variance;
        }
        continue;
#endif
        float alpha = 2.0f;
        float gamma = 0.0f;
        float alpha_candidate = 0.0f;
        float gamma_candidate = 0.0f;

        #pragma unroll
        for (int iteration = 0; iteration < EP_ITERATIONS; ++iteration) {
            const float denominator = fmaxf(
                1.0f - marginal_variance * alpha, EPS);
            const float cavity_variance = fmaxf(
                marginal_variance / denominator, EPS);
            const float cavity_mean = cavity_variance
                * (mean / marginal_variance - gamma);
            float denoised_mean, denoised_variance;
            pam4_ep_moments(cavity_mean, cavity_variance,
                denoised_mean, denoised_variance);
            const float next_alpha =
                1.0f / denoised_variance - 1.0f / cavity_variance;
            const float next_gamma = denoised_mean / denoised_variance
                - cavity_mean / cavity_variance;
            if (next_alpha > VAR_FLOOR) {
                alpha_candidate = next_alpha;
                gamma_candidate = next_gamma;
            }
            alpha = 0.5f * (alpha + alpha_candidate);
            gamma = 0.5f * (gamma + gamma_candidate);

            float lower[NR];
            #pragma unroll
            for (int j = 0; j < NR; ++j) lower[j] = 0.0f;
            float own_inverse_pivot = 0.0f;
            #pragma unroll
            for (int column = 0; column < NR; ++column) {
                float sum = 0.0f;
                #pragma unroll
                for (int previous = 0; previous < column; ++previous)
                    sum = fmaf(lower[previous],
                        tile16_shuffle(lower[previous], column), sum);
                float inverse_pivot = 0.0f;
                if (coordinate == column) {
                    lower[column] = sqrtf(fmaxf(
                        gram_lower[column] + alpha - sum, EPS));
                    inverse_pivot = 1.0f / lower[column];
                    own_inverse_pivot = inverse_pivot;
                }
                inverse_pivot = tile16_shuffle(inverse_pivot, column);
                if (coordinate > column)
                    lower[column] = (gram_lower[column] - sum) * inverse_pivot;
            }
            float forward = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row) {
                float value = matched + gamma;
                #pragma unroll
                for (int column = 0; column < row; ++column)
                    value = fmaf(-lower[column],
                        tile16_shuffle(forward, column), value);
                if (coordinate == row) forward = value * own_inverse_pivot;
            }
            mean = 0.0f;
            #pragma unroll
            for (int row = NR - 1; row >= 0; --row) {
                float value = forward;
                #pragma unroll
                for (int source = row + 1; source < NR; ++source)
                    value = fmaf(
                        -tile16_shuffle(lower[row], source),
                        tile16_shuffle(mean, source), value);
                if (coordinate == row) mean = value * own_inverse_pivot;
            }
            float inverse_column[NR];
            #pragma unroll
            for (int j = 0; j < NR; ++j) inverse_column[j] = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row) {
                const float inverse_diagonal = tile16_shuffle(
                    own_inverse_pivot, row);
                float sum = 0.0f;
                #pragma unroll
                for (int inner = 0; inner < row; ++inner) {
                    const float coefficient = tile16_shuffle(lower[inner], row);
                    if (inner >= coordinate)
                        sum = fmaf(coefficient, inverse_column[inner], sum);
                }
                if (coordinate == row)
                    inverse_column[row] = inverse_diagonal;
                else if (coordinate < row)
                    inverse_column[row] = -sum * inverse_diagonal;
            }
            marginal_variance = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row)
                if (row >= coordinate)
                    marginal_variance = fmaf(
                        inverse_column[row], inverse_column[row],
                        marginal_variance);
        }
        if (active) output[batch * NR + coordinate] = mean;
    }
}

// Fixed-H EP: the initial inverse is precomputed outside the timed region.
// Compact complex H^T/sigma^2 and P0^-1 halve the initial matched-filter and
// matrix-vector traffic, while the adaptive real Cholesky recurrence is kept
// bit-for-bit in the same 16-lane representation as the established kernel.
template<int Threads>
__global__ void fixed_h_compact_ep_kernel(
    const float* __restrict__ h_compact_scaled,
    const float* __restrict__ gram_transposed,
    const float* __restrict__ p0_inverse_compact,
    const float* __restrict__ p0_diagonal,
    const float* __restrict__ y,
    float* __restrict__ output,
    int batch_size,
    int persistent_blocks) {

    const int grouped_channel = static_cast<int>(blockIdx.y);
    h_compact_scaled += grouped_channel * MC * NC * 2;
    gram_transposed += grouped_channel * NR * NR;
    p0_inverse_compact += grouped_channel * NC * NC * 2;
    p0_diagonal += grouped_channel * NC;
    y += grouped_channel * batch_size * MR;
    output += grouped_channel * batch_size * NR;
    __shared__ float cache[648];
    float* const sh_h = cache;
    float* const sh_gram = sh_h + MC * NC * 2;
    float* const sh_p0 = sh_gram + NR * NR;
    float* const sh_diag = sh_p0 + NC * NC * 2;
    for (int index = threadIdx.x; index < 648; index += Threads) {
        if (index < 256) cache[index] = h_compact_scaled[index];
        else if (index < 512) cache[index] = gram_transposed[index - 256];
        else if (index < 640) cache[index] = p0_inverse_compact[index - 512];
        else cache[index] = p0_diagonal[index - 640];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile_in_warp = lane >> 4;
    const int coordinate = lane & 15;
    constexpr int SLOTS = Threads / NR;
    const int sample_slot = warp * 2 + tile_in_warp;
    for (int tile = blockIdx.x * SLOTS; tile < batch_size;
         tile += persistent_blocks * SLOTS) {
        const int batch = tile + sample_slot;
        const bool active = batch < batch_size;
        Pair matched_pair = pzero();
        Pair initial_mean_pair = pzero();
        if (coordinate < NC) {
            #pragma unroll
            for (int observation = 0; observation < MC; ++observation) {
                const int h_index = (observation * NC + coordinate) * 2;
                const float h_r = sh_h[h_index];
                const float h_i = sh_h[h_index + 1];
                const float y_r = active ? y[batch * MR + observation] : 0.0f;
                const float y_i = active ? y[batch * MR + observation + MC] : 0.0f;
                matched_pair.r = fmaf(h_r, y_r,
                    fmaf(h_i, y_i, matched_pair.r));
                matched_pair.i = fmaf(h_r, y_i,
                    fmaf(-h_i, y_r, matched_pair.i));
            }
        }
        #pragma unroll
        for (int column = 0; column < NC; ++column) {
            const Pair rhs = tile16_shuffle_pair(matched_pair, column);
            Pair coefficient = pzero();
            if (coordinate < NC) {
                const int matrix_index = (coordinate * NC + column) * 2;
                coefficient = Pair{
                    sh_p0[matrix_index], sh_p0[matrix_index + 1]};
            }
            initial_mean_pair.r = fmaf(coefficient.r, rhs.r,
                fmaf(-coefficient.i, rhs.i, initial_mean_pair.r));
            initial_mean_pair.i = fmaf(coefficient.r, rhs.i,
                fmaf( coefficient.i, rhs.r, initial_mean_pair.i));
        }
        const float matched_i = tile16_shuffle(matched_pair.i, coordinate & 7);
        const float mean_i = tile16_shuffle(initial_mean_pair.i, coordinate & 7);
        const float matched = coordinate < NC ? matched_pair.r : matched_i;
        float mean = coordinate < NC ? initial_mean_pair.r : mean_i;
        float marginal_variance = sh_diag[coordinate & 7];
        float alpha = 2.0f, gamma = 0.0f;
        float alpha_candidate = 0.0f, gamma_candidate = 0.0f;

        #pragma unroll
        for (int iteration = 0; iteration < EP_ITERATIONS; ++iteration) {
            const float denominator = fmaxf(
                1.0f - marginal_variance * alpha, EPS);
            const float cavity_variance = fmaxf(
                marginal_variance / denominator, EPS);
            const float cavity_mean = cavity_variance
                * (mean / marginal_variance - gamma);
            float denoised_mean, denoised_variance;
            pam4_ep_moments(cavity_mean, cavity_variance,
                denoised_mean, denoised_variance);
            const float next_alpha =
                1.0f / denoised_variance - 1.0f / cavity_variance;
            const float next_gamma = denoised_mean / denoised_variance
                - cavity_mean / cavity_variance;
            if (next_alpha > VAR_FLOOR) {
                alpha_candidate = next_alpha;
                gamma_candidate = next_gamma;
            }
            alpha = 0.5f * (alpha + alpha_candidate);
            gamma = 0.5f * (gamma + gamma_candidate);
            float lower[NR];
            #pragma unroll
            for (int j = 0; j < NR; ++j) lower[j] = 0.0f;
            float own_inverse_pivot = 0.0f;
            #pragma unroll
            for (int column = 0; column < NR; ++column) {
                float sum = 0.0f;
                #pragma unroll
                for (int previous = 0; previous < column; ++previous)
                    sum = fmaf(lower[previous],
                        tile16_shuffle(lower[previous], column), sum);
                float inverse_pivot = 0.0f;
                if (coordinate == column) {
                    lower[column] = sqrtf(fmaxf(
                        sh_gram[column * NR + coordinate] + alpha - sum, EPS));
                    inverse_pivot = 1.0f / lower[column];
                    own_inverse_pivot = inverse_pivot;
                }
                inverse_pivot = tile16_shuffle(inverse_pivot, column);
                if (coordinate > column)
                    lower[column] =
                        (sh_gram[column * NR + coordinate] - sum) * inverse_pivot;
            }
            float forward = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row) {
                float value = matched + gamma;
                #pragma unroll
                for (int column = 0; column < row; ++column)
                    value = fmaf(-lower[column],
                        tile16_shuffle(forward, column), value);
                if (coordinate == row) forward = value * own_inverse_pivot;
            }
            mean = 0.0f;
            #pragma unroll
            for (int row = NR - 1; row >= 0; --row) {
                float value = forward;
                #pragma unroll
                for (int source = row + 1; source < NR; ++source)
                    value = fmaf(-tile16_shuffle(lower[row], source),
                        tile16_shuffle(mean, source), value);
                if (coordinate == row) mean = value * own_inverse_pivot;
            }
            float inverse_column[NR];
            #pragma unroll
            for (int j = 0; j < NR; ++j) inverse_column[j] = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row) {
                const float inverse_diagonal = tile16_shuffle(
                    own_inverse_pivot, row);
                float sum = 0.0f;
                #pragma unroll
                for (int inner = 0; inner < row; ++inner) {
                    const float coefficient = tile16_shuffle(lower[inner], row);
                    if (inner >= coordinate)
                        sum = fmaf(coefficient, inverse_column[inner], sum);
                }
                if (coordinate == row)
                    inverse_column[row] = inverse_diagonal;
                else if (coordinate < row)
                    inverse_column[row] = -sum * inverse_diagonal;
            }
            marginal_variance = 0.0f;
            #pragma unroll
            for (int row = 0; row < NR; ++row)
                if (row >= coordinate)
                    marginal_variance = fmaf(inverse_column[row],
                        inverse_column[row], marginal_variance);
        }
        if (active) output[batch * NR + coordinate] = mean;
    }
}

template<int Threads, int Updates, bool Fixed, bool PackedPerH = false>
int launch_amp(
    const float* h, const float* gram, const float* y, const float* noise,
    float* output, int batch, int blocks, cudaStream_t stream) {
    if (!h || !y || !noise || !output || (Fixed && !gram)
        || batch < 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    complex_amp_kernel<Threads, Updates, Fixed, PackedPerH>
        <<<blocks, Threads, 0, stream>>>(
        h, gram, y, noise, output, batch, blocks);
    return static_cast<int>(cudaGetLastError());
}

template<int Threads>
int launch_per_h_ep(
    const float* h, const float* y, const float* noise, float* output,
    int batch, int blocks, cudaStream_t stream) {
    if (!h || !y || !noise || !output || batch < 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    per_h_hybrid_ep_kernel<Threads><<<blocks, Threads, 0, stream>>>(
        h, y, noise, output, batch, blocks);
    return static_cast<int>(cudaGetLastError());
}

template<int Threads>
int launch_fixed_ep(
    const float* h_compact, const float* gram, const float* p0,
    const float* diagonal, const float* y, float* output,
    int batch, int blocks, cudaStream_t stream) {
    if (!h_compact || !gram || !p0 || !diagonal || !y || !output
        || batch < 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    fixed_h_compact_ep_kernel<Threads><<<blocks, Threads, 0, stream>>>(
        h_compact, gram, p0, diagonal, y, output, batch, blocks);
    return static_cast<int>(cudaGetLastError());
}



// -------------------------------------------------------------------------
// Exact preprocessing shared by M vectors that use the same physical H.
// One eight-lane subgroup owns one channel.  The compact input is
// [P,16,8,2] and all workspaces are group-major.
// -------------------------------------------------------------------------


__global__ void grouped_amp_preprocess_kernel(
    const float* __restrict__ h_raw, const float* __restrict__ noise,
    float* __restrict__ h_scaled, float* __restrict__ gram_scaled,
    float* __restrict__ n0_scaled, int channels) {
    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= channels) return;
    const int tid = static_cast<int>(threadIdx.x);
    const int row = tid >> 3;
    const int column = tid & 7;
    const Pair* raw = reinterpret_cast<const Pair*>(h_raw);
    Pair* scaled = reinterpret_cast<Pair*>(h_scaled);
    Pair* gram = reinterpret_cast<Pair*>(gram_scaled);
    __shared__ Pair channel_h[MC * NC];
    #pragma unroll
    for (int linear = tid; linear < MC * NC; linear += 64) {
        const Pair value = raw[channel * MC * NC + linear];
        channel_h[linear] = value;
        scaled[channel * MC * NC + linear] =
            Pair{0.25f * value.r, 0.25f * value.i};
    }
    __syncthreads();
    Pair value = pzero();
    #pragma unroll
    for (int observation = 0; observation < MC; ++observation) {
        const Pair left = channel_h[observation * NC + row];
        const Pair right = channel_h[observation * NC + column];
        value.r = fmaf(left.r, right.r,
            fmaf(left.i, right.i, value.r));
        value.i = fmaf(left.r, right.i,
            fmaf(-left.i, right.r, value.i));
    }
    gram[(channel * NC + row) * NC + column] =
        Pair{value.r * 0.0625f, value.i * 0.0625f};
    if (tid == 0)
        n0_scaled[channel] = fmaxf(noise[channel] * 0.0625f, EPS);
}


__global__ void grouped_ep_preprocess_kernel(
    const float* __restrict__ h_raw, const float* __restrict__ noise,
    float* __restrict__ h_scaled, float* __restrict__ gram_real,
    float* __restrict__ p0_inverse, float* __restrict__ p0_diagonal,
    int channels) {
    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= channels) return;
    const int tid = static_cast<int>(threadIdx.x);
    const int row = tid >> 3;
    const int column = tid & 7;
    const Pair* raw = reinterpret_cast<const Pair*>(h_raw);
    Pair* scaled = reinterpret_cast<Pair*>(h_scaled);
    Pair* inverse = reinterpret_cast<Pair*>(p0_inverse);
    const float inv_noise = 1.0f / fmaxf(noise[channel], EPS);

    __shared__ Pair channel_h[MC * NC];
    __shared__ Pair gram[NC * NC];
    __shared__ Pair lower[NC * NC];
    __shared__ float inverse_pivot[NC];

    #pragma unroll
    for (int linear = tid; linear < MC * NC; linear += 64) {
        const Pair value = raw[channel * MC * NC + linear];
        channel_h[linear] = value;
        scaled[channel * MC * NC + linear] =
            Pair{value.r * inv_noise, value.i * inv_noise};
    }
    __syncthreads();

    Pair value = pzero();
    #pragma unroll
    for (int observation = 0; observation < MC; ++observation) {
        const Pair left = channel_h[observation * NC + row];
        const Pair right = channel_h[observation * NC + column];
        value.r = fmaf(left.r, right.r,
            fmaf(left.i, right.i, value.r));
        value.i = fmaf(left.r, right.i,
            fmaf(-left.i, right.r, value.i));
    }
    value.r *= inv_noise;
    value.i *= inv_noise;
    gram[row * NC + column] = value;
    lower[row * NC + column] = pzero();

    // Store the exact 16x16 real-equivalent Gram expected by EP.
    gram_real[(channel * NR + row) * NR + column] = value.r;
    gram_real[(channel * NR + row) * NR + column + NC] = -value.i;
    gram_real[(channel * NR + row + NC) * NR + column] = value.i;
    gram_real[(channel * NR + row + NC) * NR + column + NC] = value.r;
    __syncthreads();

    // Exact Cholesky of G + 2I with the original arithmetic order.
    #pragma unroll
    for (int pivot = 0; pivot < NC; ++pivot) {
        if (row == pivot && column == pivot) {
            Pair sum = pzero();
            #pragma unroll
            for (int previous = 0; previous < pivot; ++previous) {
                const Pair own = lower[pivot * NC + previous];
                sum.r = fmaf(own.r, own.r, fmaf(own.i, own.i, sum.r));
                sum.i = fmaf(own.i, own.r, fmaf(-own.r, own.i, sum.i));
            }
            const float diagonal = sqrtf(fmaxf(
                gram[pivot * NC + pivot].r + 2.0f - sum.r, EPS));
            lower[pivot * NC + pivot] = Pair{diagonal, 0.0f};
            inverse_pivot[pivot] = 1.0f / diagonal;
        }
        __syncthreads();
        if (column == pivot && row > pivot) {
            Pair sum = pzero();
            #pragma unroll
            for (int previous = 0; previous < pivot; ++previous) {
                const Pair own = lower[row * NC + previous];
                const Pair other = lower[pivot * NC + previous];
                sum.r = fmaf(own.r, other.r,
                    fmaf(own.i, other.i, sum.r));
                sum.i = fmaf(own.i, other.r,
                    fmaf(-own.r, other.i, sum.i));
            }
            const Pair entry = gram[row * NC + pivot];
            const float inv = inverse_pivot[pivot];
            lower[row * NC + pivot] =
                Pair{(entry.r - sum.r) * inv, (entry.i - sum.i) * inv};
        }
        __syncthreads();
    }

    // Eight exact right-hand sides are independent after factorization.
    const int rhs = tid >> 3;
    const int coordinate = tid & 7;
    const int warp_subgroup = (tid & 31) >> 3;
    const unsigned mask = 0xffu << (warp_subgroup * 8);
    Pair forward = pzero();
    #pragma unroll
    for (int solve_row = 0; solve_row < NC; ++solve_row) {
        Pair solve_value{
            coordinate == solve_row && solve_row == rhs ? 1.0f : 0.0f,
            0.0f};
        #pragma unroll
        for (int source = 0; source < solve_row; ++source) {
            const Pair solved{
                __shfl_sync(mask, forward.r, source, NC),
                __shfl_sync(mask, forward.i, source, NC)};
            const Pair coefficient = lower[coordinate * NC + source];
            solve_value.r = fmaf(-coefficient.r, solved.r,
                fmaf(coefficient.i, solved.i, solve_value.r));
            solve_value.i = fmaf(-coefficient.r, solved.i,
                fmaf(-coefficient.i, solved.r, solve_value.i));
        }
        if (coordinate == solve_row) {
            forward.r = solve_value.r * inverse_pivot[coordinate];
            forward.i = solve_value.i * inverse_pivot[coordinate];
        }
    }
    Pair solution = pzero();
    #pragma unroll
    for (int solve_row = NC - 1; solve_row >= 0; --solve_row) {
        Pair solve_value = forward;
        #pragma unroll
        for (int source = solve_row + 1; source < NC; ++source) {
            const Pair raw_coefficient = lower[source * NC + solve_row];
            const Pair coefficient{raw_coefficient.r, -raw_coefficient.i};
            const Pair solved{
                __shfl_sync(mask, solution.r, source, NC),
                __shfl_sync(mask, solution.i, source, NC)};
            solve_value.r = fmaf(-coefficient.r, solved.r,
                fmaf(coefficient.i, solved.i, solve_value.r));
            solve_value.i = fmaf(-coefficient.r, solved.i,
                fmaf(-coefficient.i, solved.r, solve_value.i));
        }
        if (coordinate == solve_row) {
            solution.r = solve_value.r * inverse_pivot[coordinate];
            solution.i = solve_value.i * inverse_pivot[coordinate];
        }
    }
    inverse[(channel * NC + coordinate) * NC + rhs] = solution;
    if (rhs == coordinate)
        p0_diagonal[channel * NC + coordinate] = solution.r;
}

template<int Threads, int Updates>
int launch_grouped_amp_impl(
    bool preprocess, const float* h, const float* y, const float* noise,
    float* output, float* h_scaled, float* gram, float* n0,
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream) {
    if (!h || !y || !noise || !output || !h_scaled || !gram || !n0 ||
        channels <= 0 || vectors_per_channel <= 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (preprocess) {
        grouped_amp_preprocess_kernel<<<channels, 64, 0, stream>>>(
            h, noise, h_scaled, gram, n0, channels);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    complex_amp_kernel<Threads, Updates, true, false>
        <<<dim3(blocks, channels), Threads, 0, stream>>>(
            h_scaled, gram, y, n0, output, vectors_per_channel, blocks);
    return static_cast<int>(cudaGetLastError());
}

template<int Threads>
int launch_grouped_ep_impl(
    bool preprocess, const float* h, const float* y, const float* noise,
    float* output, float* h_scaled, float* gram, float* p0, float* diagonal,
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream) {
    if (!h || !y || !noise || !output || !h_scaled || !gram || !p0 ||
        !diagonal || channels <= 0 || vectors_per_channel <= 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (preprocess) {
        grouped_ep_preprocess_kernel<<<channels, 64, 0, stream>>>(
            h, noise, h_scaled, gram, p0, diagonal, channels);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    fixed_h_compact_ep_kernel<Threads>
        <<<dim3(blocks, channels), Threads, 0, stream>>>(
            h_scaled, gram, p0, diagonal, y, output,
            vectors_per_channel, blocks);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

#define GROUPED_SIGNATURE(NAME) \
extern "C" int NAME( \
    int task, const float* h, const float* y, const float* noise, float* output, \
    float* w0, float* w1, float* w2, float* w3, float* w4, float* w5, \
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream)

#define DEFINE_GROUPED_AMP(T, U, I) \
GROUPED_SIGNATURE(launch_grouped_amp_t##I##_##T##_full) { \
    (void)task; (void)w3; (void)w4; (void)w5; \
    return launch_grouped_amp_impl<T, U>(true, h, y, noise, output, w0, w1, w2, \
        channels, vectors_per_channel, blocks, stream); \
} \
GROUPED_SIGNATURE(launch_grouped_amp_t##I##_##T##_core) { \
    (void)task; (void)w3; (void)w4; (void)w5; \
    return launch_grouped_amp_impl<T, U>(false, h, y, noise, output, w0, w1, w2, \
        channels, vectors_per_channel, blocks, stream); \
}

#define DEFINE_GROUPED_EP(T) \
GROUPED_SIGNATURE(launch_grouped_ep_##T##_full) { \
    (void)task; (void)w4; (void)w5; \
    return launch_grouped_ep_impl<T>(true, h, y, noise, output, w0, w1, w2, w3, \
        channels, vectors_per_channel, blocks, stream); \
} \
GROUPED_SIGNATURE(launch_grouped_ep_##T##_core) { \
    (void)task; (void)w4; (void)w5; \
    return launch_grouped_ep_impl<T>(false, h, y, noise, output, w0, w1, w2, w3, \
        channels, vectors_per_channel, blocks, stream); \
}

DEFINE_GROUPED_AMP(64, 7, 8)
DEFINE_GROUPED_AMP(128, 7, 8)
DEFINE_GROUPED_AMP(64, 14, 15)
DEFINE_GROUPED_AMP(128, 14, 15)
DEFINE_GROUPED_EP(64)
DEFINE_GROUPED_EP(96)
DEFINE_GROUPED_EP(128)
