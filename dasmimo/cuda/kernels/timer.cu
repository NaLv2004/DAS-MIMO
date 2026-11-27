#include <cuda_runtime.h>

// Every grouped-H detector uses the same ABI.  The six workspaces are
// preallocated by the caller; allocation and host/device copies therefore do
// not leak into the detector timing boundary.  A launcher may ignore any
// workspace it does not need.
using GroupedLaunch = int (*)(
    int, const float*, const float*, const float*, float*,
    float*, float*, float*, float*, float*, float*,
    int, int, int, cudaStream_t);

namespace {

int begin_timing(cudaStream_t stream, cudaEvent_t* start, cudaEvent_t* stop) {
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return static_cast<int>(error);
    error = cudaEventCreate(start);
    if (error != cudaSuccess) return static_cast<int>(error);
    error = cudaEventCreate(stop);
    if (error != cudaSuccess) {
        cudaEventDestroy(*start);
        return static_cast<int>(error);
    }
    return static_cast<int>(cudaEventRecord(*start, stream));
}

int finish_timing(
    cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
    int replays, float* average_ms) {
    cudaError_t error = cudaEventRecord(stop, stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    if (error == cudaSuccess)
        error = cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (error != cudaSuccess) return static_cast<int>(error);
    *average_ms = elapsed_ms / static_cast<float>(replays);
    return 0;
}

}  // namespace

extern "C" int benchmark_grouped_h_launch_loop(
    void* function_pointer, int task,
    const float* h, const float* y, const float* noise, float* output,
    float* workspace0, float* workspace1, float* workspace2,
    float* workspace3, float* workspace4, float* workspace5,
    int channels, int vectors_per_channel, int blocks_per_channel,
    int warmups, int replays, cudaStream_t stream, float* average_ms) {
    if (!function_pointer || !h || !y || !noise || !output ||
        !average_ms || channels <= 0 || vectors_per_channel <= 0 ||
        blocks_per_channel <= 0 || warmups < 0 || replays <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const GroupedLaunch launch =
        reinterpret_cast<GroupedLaunch>(function_pointer);
    for (int index = 0; index < warmups; ++index) {
        const int status = launch(
            task, h, y, noise, output,
            workspace0, workspace1, workspace2,
            workspace3, workspace4, workspace5,
            channels, vectors_per_channel, blocks_per_channel, stream);
        if (status != 0) return status;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    int status = begin_timing(stream, &start, &stop);
    if (status != 0) return status;
    for (int index = 0; index < replays; ++index) {
        status = launch(
            task, h, y, noise, output,
            workspace0, workspace1, workspace2,
            workspace3, workspace4, workspace5,
            channels, vectors_per_channel, blocks_per_channel, stream);
        if (status != 0) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return status;
        }
    }
    return finish_timing(
        stream, start, stop, replays, average_ms);
}
