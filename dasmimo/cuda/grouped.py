import re

PREPROCESS_PARALLEL_SOURCE = r"""

// Latency-oriented exact preprocessor: one 64-thread block per physical H.
// The 64 Gram entries and the eight A^{-1} right-hand sides are evaluated in
// parallel.  The Cholesky arithmetic and solve order match the grouped
// reference kernel; only independent work is exposed concurrently.
__global__ void grouped_dag_preprocess_kernel(
    const Pair* __restrict__ h_raw, const float* __restrict__ noise,
    Pair* __restrict__ h_normalized, Pair* __restrict__ a_normalized,
    float* __restrict__ d_inverse, Pair* __restrict__ a_inverse,
    float* __restrict__ sigma_normalized, int channels) {
    constexpr int NC = 8, MC = 16;
    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= channels) return;
    const int tid = static_cast<int>(threadIdx.x);
    const int row = tid >> 3;
    const int column = tid & 7;
    const float physical_sigma = fmaxf(noise[channel], 1.0e-12f);

    __shared__ Pair gram[NC * NC];
    __shared__ Pair lower[NC * NC];
    __shared__ float inverse_pivot[NC];
    __shared__ float row_sums[NC];
    __shared__ float shared_inv_scale;
    __shared__ float shared_inv_sqrt_scale;

    Pair value = pzero();
    #pragma unroll
    for (int observation = 0; observation < MC; ++observation) {
        const Pair left = h_raw[(channel * MC + observation) * NC + row];
        const Pair right = h_raw[(channel * MC + observation) * NC + column];
        value.r = fmaf(left.r, right.r,
            fmaf(left.i, right.i, value.r));
        value.i = fmaf(left.r, right.i,
            fmaf(-left.i, right.r, value.i));
    }
    gram[row * NC + column] = value;
    lower[row * NC + column] = pzero();
    __syncthreads();

    if (column == 0) {
        float row_sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Pair entry = gram[row * NC + c];
            const float real_value = entry.r + (c == row ? physical_sigma : 0.0f);
            row_sum += fabsf(real_value) + fabsf(entry.i);
        }
        row_sums[row] = row_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float scale = row_sums[0];
        #pragma unroll
        for (int r = 1; r < NC; ++r) scale = fmaxf(scale, row_sums[r]);
        scale = fmaxf(scale, 1.0e-12f);
        shared_inv_scale = 1.0f / scale;
        shared_inv_sqrt_scale = rsqrtf(scale);
        sigma_normalized[channel * 2] = physical_sigma * shared_inv_scale;
        sigma_normalized[channel * 2 + 1] = shared_inv_sqrt_scale;
    }
    __syncthreads();
    const float inv_scale = shared_inv_scale;
    const float inv_sqrt_scale = shared_inv_sqrt_scale;

    value = gram[row * NC + column];
    value.r *= inv_scale;
    value.i *= inv_scale;
    if (row == column) value.r += physical_sigma * inv_scale;
    a_normalized[(channel * NC + row) * NC + column] = value;
    if (column == 0) {
        const float diagonal =
            (gram[row * NC + row].r + physical_sigma) * inv_scale;
        d_inverse[channel * NC + row] = 1.0f / (diagonal + 1.0e-6f);
    }
    #pragma unroll
    for (int linear = tid; linear < MC * NC; linear += 64) {
        const int observation = linear >> 3;
        const int coordinate = linear & 7;
        const Pair h = h_raw[(channel * MC + observation) * NC + coordinate];
        h_normalized[(channel * MC + observation) * NC + coordinate] =
            Pair{h.r * inv_sqrt_scale, h.i * inv_sqrt_scale};
    }

    // Exact Hermitian Cholesky factorization.  One thread forms each live
    // element, with a block barrier preserving the column dependencies.
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
                (gram[pivot * NC + pivot].r + physical_sigma) * inv_scale
                    - sum.r,
                1.0e-12f));
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
                Pair{(entry.r * inv_scale - sum.r) * inv,
                     (entry.i * inv_scale - sum.i) * inv};
        }
        __syncthreads();
    }

    // Eight independent 8-lane subgroups solve A X = I concurrently.
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
    a_inverse[(channel * NC + coordinate) * NC + rhs] = solution;
}
"""

KERNEL_RE = re.compile(
    r"(__global__\s+__launch_bounds__\(\d+\)\s+void\s+"
    r"(?P<name>dag_fixed_complex(?:_pack[24])?_task_(?P<task>\d+))\(.*?"
    r"float\*\s+__restrict__\s+output,\s+int batch_size,\s+"
    r"int persistent_blocks\)\s*\{)",
    re.S,
)


def transform(source: str, task: int, aligned: bool, warps: int) -> str:
    mc, nc = 16, 8
    kernels = source
    insertion = f"""
    const int grouped_channel = static_cast<int>(blockIdx.y);
    H_input += grouped_channel * {mc} * {nc};
    y += grouped_channel * batch_size * {2 * mc};
    A_input += grouped_channel * {nc} * {nc};
    D_inv_input += grouped_channel * {nc};
    Ainv_input += grouped_channel * {nc} * {nc};
    sigma2_input += grouped_channel * 2;
    const float grouped_y_scale = sigma2_input[1];
    output += grouped_channel * batch_size * {2 * nc};
"""
    kernels = KERNEL_RE.sub(lambda match: match.group(1) + insertion, kernels)
    # The fixed-H source receives an already normalized y.  In the grouped
    # full path H is raw, so fuse y/sqrt(scale) into the existing loads instead
    # of materializing another P*M*32 workspace.
    kernels = re.sub(
        r"(?<![A-Za-z0-9_])y\[([^\]]+)\]",
        r"(grouped_y_scale * y[\1])",
        kernels,
    )
    kernels = re.sub(
        r"(?<![A-Za-z0-9_])(y_(?:real|imag)\d*\.[xyzw])",
        r"(grouped_y_scale * \1)",
        kernels,
    )
    switch = (
        f"        case {task}: dag_fixed_complex_pack2_task_{task:02d}<<<dim3(blocks, channels), {32 * warps}, 0, stream>>>("
        "H, y, A, D_inv, A_inv, sigma2, output, vectors_per_channel, blocks); break;"
    )
    preprocess_source = PREPROCESS_PARALLEL_SOURCE
    preprocess_blocks = "channels"
    preprocess_launch_threads = 64
    core_h = "reinterpret_cast<const Pair*>(w0)"
    launcher = f"""
int launch_grouped_dag_core_impl(
    int task, const Pair* H, const float* y, const Pair* A,
    const float* D_inv, const Pair* A_inv, const float* sigma2,
    float* output, int channels, int vectors_per_channel, int blocks,
    cudaStream_t stream) {{
    if (!H || !y || !A || !D_inv || !A_inv || !sigma2 || !output ||
        channels <= 0 || vectors_per_channel <= 0 || blocks <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    {f"if (vectors_per_channel % {8 * warps}) return -2;" if aligned else ""}
    switch (task) {{
{switch}
        default: return -1;
    }}
    return static_cast<int>(cudaGetLastError());
}}

#include <mutex>
struct GroupedDagPipelineState {{
    std::once_flag once;
    cudaError_t init_error = cudaSuccess;
    cudaStream_t streams[4] = {{nullptr, nullptr, nullptr, nullptr}};
    cudaStream_t capture_stream = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t done[4] = {{nullptr, nullptr, nullptr, nullptr}};
}};
static GroupedDagPipelineState grouped_dag_pipeline_state;

static void initialize_grouped_dag_pipeline() {{
    auto& state = grouped_dag_pipeline_state;
    std::call_once(state.once, [&state]() {{
        state.init_error = cudaEventCreateWithFlags(
            &state.ready, cudaEventDisableTiming);
        if (state.init_error == cudaSuccess)
            state.init_error = cudaStreamCreateWithFlags(
                &state.capture_stream, cudaStreamNonBlocking);
        for (int index = 0; index < 4 && state.init_error == cudaSuccess; ++index) {{
            state.init_error = cudaStreamCreateWithFlags(
                &state.streams[index], cudaStreamNonBlocking);
            if (state.init_error == cudaSuccess)
                state.init_error = cudaEventCreateWithFlags(
                    &state.done[index], cudaEventDisableTiming);
        }}
    }});
}}

template<int PipelineStreams, int ChannelChunk>
int enqueue_grouped_dag_full_pipelined(
    int task, const float* h, const float* y, const float* noise,
    float* output, float* w0, float* w1, float* w2, float* w3,
    float* w4, int channels, int vectors_per_channel, int blocks,
    cudaStream_t caller_stream) {{
    static_assert(PipelineStreams == 2 || PipelineStreams == 4,
        "supported pipeline widths are two/four streams");
    initialize_grouped_dag_pipeline();
    auto& state = grouped_dag_pipeline_state;
    if (state.init_error != cudaSuccess)
        return static_cast<int>(state.init_error);
    cudaError_t error = cudaEventRecord(state.ready, caller_stream);
    if (error != cudaSuccess) return static_cast<int>(error);
    bool used[PipelineStreams] = {{}};
    int sequence = 0;
    for (int first = 0; first < channels; first += ChannelChunk, ++sequence) {{
        const int count = first + ChannelChunk <= channels
            ? ChannelChunk : channels - first;
        const int stream_index = sequence % PipelineStreams;
        cudaStream_t work_stream = state.streams[stream_index];
        if (!used[stream_index]) {{
            error = cudaStreamWaitEvent(work_stream, state.ready, 0);
            if (error != cudaSuccess) return static_cast<int>(error);
            used[stream_index] = true;
        }}
        const float* h_part = h + first * {2 * mc * nc};
        const float* y_part = y + first * vectors_per_channel * {2 * mc};
        const float* noise_part = noise + first;
        float* output_part = output + first * vectors_per_channel * {2 * nc};
        float* w0_part = w0 + first * {2 * mc * nc};
        float* w1_part = w1 + first * {2 * nc * nc};
        float* w2_part = w2 + first * {nc};
        float* w3_part = w3 + first * {2 * nc * nc};
        float* w4_part = w4 + first * 2;
        grouped_dag_preprocess_kernel<<<
            count, {preprocess_launch_threads}, 0, work_stream>>>(
            reinterpret_cast<const Pair*>(h_part), noise_part,
            reinterpret_cast<Pair*>(w0_part),
            reinterpret_cast<Pair*>(w1_part), w2_part,
            reinterpret_cast<Pair*>(w3_part), w4_part, count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        const int status = launch_grouped_dag_core_impl(
            task, reinterpret_cast<const Pair*>(w0_part),
            y_part, reinterpret_cast<const Pair*>(w1_part), w2_part,
            reinterpret_cast<const Pair*>(w3_part), w4_part, output_part,
            count, vectors_per_channel, blocks, work_stream);
        if (status != 0) return status;
    }}
    for (int index = 0; index < PipelineStreams; ++index) {{
        if (!used[index]) continue;
        error = cudaEventRecord(state.done[index], state.streams[index]);
        if (error != cudaSuccess) return static_cast<int>(error);
        error = cudaStreamWaitEvent(caller_stream, state.done[index], 0);
        if (error != cudaSuccess) return static_cast<int>(error);
    }}
    return 0;
}}

struct GroupedDagGraphKey {{
    const void* pointers[10] = {{}};
    int channels = 0;
    int vectors = 0;
    int blocks = 0;
}};

struct GroupedDagGraphSlot {{
    cudaGraphExec_t executable = nullptr;
    GroupedDagGraphKey key;
}};

template<int PipelineStreams, int ChannelChunk>
struct GroupedDagGraphCache {{
    std::mutex mutex;
    GroupedDagGraphSlot slots[24];
}};

static bool grouped_dag_graph_key_equal(
    const GroupedDagGraphKey& left, const GroupedDagGraphKey& right) {{
    if (left.channels != right.channels || left.vectors != right.vectors ||
        left.blocks != right.blocks) return false;
    for (int index = 0; index < 10; ++index)
        if (left.pointers[index] != right.pointers[index]) return false;
    return true;
}}

template<int PipelineStreams, int ChannelChunk>
int launch_grouped_dag_full_pipelined(
    int task, const float* h, const float* y, const float* noise,
    float* output, float* w0, float* w1, float* w2, float* w3,
    float* w4, int channels, int vectors_per_channel, int blocks,
    cudaStream_t caller_stream) {{
    if (task < 0 || task >= 24) return -1;
    initialize_grouped_dag_pipeline();
    auto& state = grouped_dag_pipeline_state;
    if (state.init_error != cudaSuccess)
        return static_cast<int>(state.init_error);
    static GroupedDagGraphCache<PipelineStreams, ChannelChunk> cache;
    GroupedDagGraphKey requested;
    requested.pointers[0] = h; requested.pointers[1] = y;
    requested.pointers[2] = noise; requested.pointers[3] = output;
    requested.pointers[4] = w0; requested.pointers[5] = w1;
    requested.pointers[6] = w2; requested.pointers[7] = w3;
    requested.pointers[8] = w4; requested.pointers[9] = nullptr;
    requested.channels = channels;
    requested.vectors = vectors_per_channel;
    requested.blocks = blocks;
    std::lock_guard<std::mutex> guard(cache.mutex);
    auto& slot = cache.slots[task];
    if (!slot.executable || !grouped_dag_graph_key_equal(slot.key, requested)) {{
        if (slot.executable) {{
            cudaGraphExecDestroy(slot.executable);
            slot.executable = nullptr;
        }}
        cudaError_t error = cudaStreamBeginCapture(
            state.capture_stream, cudaStreamCaptureModeGlobal);
        if (error != cudaSuccess) return static_cast<int>(error);
        const int enqueue_status = enqueue_grouped_dag_full_pipelined<
            PipelineStreams, ChannelChunk>(
            task, h, y, noise, output, w0, w1, w2, w3, w4,
            channels, vectors_per_channel, blocks, state.capture_stream);
        if (enqueue_status != 0) {{
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(state.capture_stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            return enqueue_status;
        }}
        cudaGraph_t graph = nullptr;
        error = cudaStreamEndCapture(state.capture_stream, &graph);
        if (error != cudaSuccess) return static_cast<int>(error);
        error = cudaGraphInstantiate(&slot.executable, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        if (error != cudaSuccess) return static_cast<int>(error);
        slot.key = requested;
    }}
    return static_cast<int>(cudaGraphLaunch(slot.executable, caller_stream));
}}

#define GROUPED_DAG_SIGNATURE(NAME) \\
extern "C" int NAME( \\
    int task, const float* h, const float* y, const float* noise, float* output, \\
    float* w0, float* w1, float* w2, float* w3, float* w4, float* w5, \\
    int channels, int vectors_per_channel, int blocks, cudaStream_t stream)

GROUPED_DAG_SIGNATURE(launch_grouped_dag_full) {{
    (void)w5;
    if (!h || !noise || !w0 || !w1 || !w2 || !w3 || !w4)
        return static_cast<int>(cudaErrorInvalidValue);
    grouped_dag_preprocess_kernel<<<
        {preprocess_blocks}, {preprocess_launch_threads}, 0, stream>>>(
        reinterpret_cast<const Pair*>(h), noise,
        reinterpret_cast<Pair*>(w0), reinterpret_cast<Pair*>(w1), w2,
        reinterpret_cast<Pair*>(w3), w4, channels);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    return launch_grouped_dag_core_impl(
        task, {core_h}, y,
        reinterpret_cast<const Pair*>(w1), w2,
        reinterpret_cast<const Pair*>(w3), w4, output,
        channels, vectors_per_channel, blocks, stream);
}}

GROUPED_DAG_SIGNATURE(launch_grouped_dag_core) {{
    (void)noise; (void)w5;
    return launch_grouped_dag_core_impl(
        task, {core_h}, y,
        reinterpret_cast<const Pair*>(w1), w2,
        reinterpret_cast<const Pair*>(w3), w4, output,
        channels, vectors_per_channel, blocks, stream);
}}

#define GROUPED_DAG_PIPELINE_WRAPPER(STREAMS, CHUNK) \
GROUPED_DAG_SIGNATURE(launch_grouped_dag_full_pipe##STREAMS##_c##CHUNK) {{ \
    (void)w5; \
    if (!h || !noise || !w0 || !w1 || !w2 || !w3 || !w4) \
        return static_cast<int>(cudaErrorInvalidValue); \
    return launch_grouped_dag_full_pipelined<STREAMS, CHUNK>( \
        task, h, y, noise, output, w0, w1, w2, w3, w4, \
        channels, vectors_per_channel, blocks, stream); \
}}
GROUPED_DAG_PIPELINE_WRAPPER(2, 4)
GROUPED_DAG_PIPELINE_WRAPPER(2, 8)
GROUPED_DAG_PIPELINE_WRAPPER(2, 16)
GROUPED_DAG_PIPELINE_WRAPPER(2, 32)
GROUPED_DAG_PIPELINE_WRAPPER(2, 64)
GROUPED_DAG_PIPELINE_WRAPPER(2, 128)
GROUPED_DAG_PIPELINE_WRAPPER(4, 4)
GROUPED_DAG_PIPELINE_WRAPPER(4, 8)
GROUPED_DAG_PIPELINE_WRAPPER(4, 16)
GROUPED_DAG_PIPELINE_WRAPPER(4, 32)
GROUPED_DAG_PIPELINE_WRAPPER(4, 64)
GROUPED_DAG_PIPELINE_WRAPPER(4, 128)
#undef GROUPED_DAG_PIPELINE_WRAPPER
"""
    return kernels + preprocess_source + launcher
