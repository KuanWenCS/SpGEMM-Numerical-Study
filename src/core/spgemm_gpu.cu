#include "spgemm_gpu.hpp"
#include "experiment.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda/atomic>

namespace study {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count)
    {
        if (!count) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Device allocation size overflow.");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
    cudaEvent_t value;
    Event() { check_cuda(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <typename T>
void upload(const DeviceBuffer<T>& destination, const std::vector<T>& source)
{
    if (!source.empty()) check_cuda(cudaMemcpy(destination.data, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
struct DeviceCSRView {
    Index rows;
    const Offset* row_ptr;
    const Index* col_idx;
    const T* values;
};

__device__ Offset find_column(const Index* columns, Offset begin, Offset end, Index target)
{
    while (begin < end) {
        const Offset middle = begin + (end - begin) / 2;
        if (columns[middle] < target) begin = middle + 1;
        else end = middle;
    }
    return begin;
}

template <typename T>
__global__ void multiply_rows(DeviceCSRView<T> A, DeviceCSRView<T> P,
    const Offset* c_rows, const Index* c_columns, T* output, int* error)
{
    const Offset first = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Offset stride = static_cast<Offset>(gridDim.x) * blockDim.x;
    for (Offset i = first; i < A.rows; i += stride) {
        const Offset begin = c_rows[i], end = c_rows[i + 1];
        for (Offset p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            const Index k = A.col_idx[p];
            for (Offset q = P.row_ptr[k]; q < P.row_ptr[k + 1]; ++q) {
                const Index j = P.col_idx[q];
                const Offset slot = find_column(c_columns, begin, end, j);
                if (slot == end || c_columns[slot] != j) { atomicExch(error, 1); return; }
                const T product = A.values[p] * P.values[q];
                output[slot] = output[slot] + product;
            }
        }
    }
}

template <typename T>
__global__ void multiply_atomic(DeviceCSRView<T> A, DeviceCSRView<T> P,
    const Offset* c_rows, const Index* c_columns, T* output, int* error,
    Offset trace_slot, Offset capacity, int* lock, Offset* count, TraceEvent<T>* events)
{
    const Offset worker = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (Offset i = blockIdx.x; i < A.rows; i += gridDim.x) {
        const Offset begin = c_rows[i], end = c_rows[i + 1];
        for (Offset p = A.row_ptr[i] + threadIdx.x; p < A.row_ptr[i + 1]; p += blockDim.x) {
            const Index k = A.col_idx[p];
            for (Offset q = P.row_ptr[k]; q < P.row_ptr[k + 1]; ++q) {
                const Index j = P.col_idx[q];
                const Offset slot = find_column(c_columns, begin, end, j);
                if (slot == end || c_columns[slot] != j) { atomicExch(error, 1); return; }
                const T product = A.values[p] * P.values[q];
                if (slot == trace_slot) {
                    cuda::atomic_ref<int, cuda::thread_scope_device> gate(*lock);
                    int expected = 0;
                    while (!gate.compare_exchange_weak(expected, 1,
                        cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed)) {
                        expected = 0;
                        __nanosleep(64);
                    }
                    const T before = atomicAdd(output + slot, product);
                    cuda::atomic_ref<T, cuda::thread_scope_device> destination(output[slot]);
                    const T after = destination.load(cuda::std::memory_order_relaxed);
                    const Offset event = *count;
                    if (event < capacity) events[event] = {p, q, worker, product, before, after};
                    else atomicExch(error, 2);
                    *count = event + 1;
                    gate.store(0, cuda::std::memory_order_release);
                } else atomicAdd(output + slot, product);
            }
        }
    }
}

__device__ bool same_bits(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
__device__ bool same_bits(double a, double b) { return __double_as_longlong(a) == __double_as_longlong(b); }

template <typename T>
__global__ void replay_events(const TraceEvent<T>* events, Offset count, T* output, int* error)
{
    for (Offset i = 0; i < count; ++i) {
        const T before = atomicAdd(output, events[i].product);
        cuda::atomic_ref<T, cuda::thread_scope_device> destination(*output);
        const T after = destination.load(cuda::std::memory_order_relaxed);
        if (!same_bits(before, events[i].before) || !same_bits(after, events[i].after)) *error = 1;
    }
}
}

std::string gpu_description()
{
    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaGetDeviceProperties(&properties, device));
    check_cuda(cudaRuntimeGetVersion(&runtime));
    check_cuda(cudaDriverGetVersion(&driver));
    std::ostringstream out;
    out << properties.name << "; compute=" << properties.major << '.' << properties.minor
        << "; runtime=" << runtime << "; driver=" << driver;
    return out.str();
}

template <typename T>
ExecutionResult<T> execute_gpu(const CSRMatrix& A, const CSRMatrix& P,
    const NumericInputs<T>& inputs, const CSRPattern& pattern,
    Strategy strategy, const ExecutionOptions& options)
{
    const auto start = Clock::now();
    validate_csr(A);
    validate_csr(P);
    validate_output_pattern(A, P, pattern);
    if (inputs.a_values.size() != A.values.size() || inputs.p_values.size() != P.values.size()) {
        throw std::invalid_argument("Typed input lengths differ.");
    }
    if (!options.gpu_threads || options.gpu_threads > 1024) throw std::invalid_argument("Invalid GPU block size.");
    if (options.trace_slot) {
        if (strategy != Strategy::atomic || *options.trace_slot >= pattern.col_idx.size() || !options.trace_capacity) {
            throw std::invalid_argument("Invalid GPU trace selection.");
        }
        int device = 0;
        cudaDeviceProp prop{};
        check_cuda(cudaGetDevice(&device));
        check_cuda(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 7) throw std::runtime_error("Lock-based trace requires compute capability >= 7.0.");
    }
    ExecutionResult<T> result;
    result.values.assign(pattern.col_idx.size(), T{0});
    if (result.values.empty()) { result.total_seconds = seconds(start); return result; }
    DeviceBuffer<Offset> ar(A.row_ptr.size()), pr(P.row_ptr.size()), cr(pattern.row_ptr.size());
    DeviceBuffer<Index> ac(A.col_idx.size()), pc(P.col_idx.size()), cc(pattern.col_idx.size());
    DeviceBuffer<T> av(inputs.a_values.size()), pv(inputs.p_values.size()), cv(result.values.size());
    DeviceBuffer<int> error(1), lock(1);
    DeviceBuffer<Offset> count(1);
    const Offset capacity = options.trace_slot ? options.trace_capacity : 0;
    DeviceBuffer<TraceEvent<T>> events(capacity);
    upload(ar, A.row_ptr); upload(ac, A.col_idx); upload(av, inputs.a_values);
    upload(pr, P.row_ptr); upload(pc, P.col_idx); upload(pv, inputs.p_values);
    upload(cr, pattern.row_ptr); upload(cc, pattern.col_idx);
    check_cuda(cudaMemset(cv.data, 0, result.values.size() * sizeof(T)));
    check_cuda(cudaMemset(error.data, 0, sizeof(int)));
    check_cuda(cudaMemset(lock.data, 0, sizeof(int)));
    check_cuda(cudaMemset(count.data, 0, sizeof(Offset)));
    const DeviceCSRView<T> ga{A.rows, ar.data, ac.data, av.data};
    const DeviceCSRView<T> gp{P.rows, pr.data, pc.data, pv.data};
    const Offset wanted = strategy == Strategy::serial
        ? (static_cast<Offset>(A.rows) + options.gpu_threads - 1) / options.gpu_threads : A.rows;
    const unsigned blocks = static_cast<unsigned>(std::min<Offset>(wanted, 65535));
    Event begin, end;
    result.prepare_seconds = seconds(start);
    check_cuda(cudaEventRecord(begin.value));
    if (strategy == Strategy::serial) {
        multiply_rows<T><<<blocks, options.gpu_threads>>>(ga, gp, cr.data, cc.data, cv.data, error.data);
    } else {
        multiply_atomic<T><<<blocks, options.gpu_threads>>>(ga, gp, cr.data, cc.data, cv.data, error.data,
            options.trace_slot.value_or(std::numeric_limits<Offset>::max()), capacity, lock.data, count.data, events.data);
    }
    check_cuda(cudaGetLastError());
    check_cuda(cudaEventRecord(end.value));
    check_cuda(cudaEventSynchronize(end.value));
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, begin.value, end.value));
    result.compute_seconds = milliseconds / 1000.0;
    const auto download = Clock::now();
    int host_error = 0;
    check_cuda(cudaMemcpy(&host_error, error.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_error) throw std::runtime_error("GPU output pattern or trace capacity mismatch.");
    check_cuda(cudaMemcpy(result.values.data(), cv.data, result.values.size() * sizeof(T), cudaMemcpyDeviceToHost));
    if (options.trace_slot) {
        Offset trace_count = 0;
        check_cuda(cudaMemcpy(&trace_count, count.data, sizeof(Offset), cudaMemcpyDeviceToHost));
        if (trace_count > capacity) throw std::runtime_error("GPU trace overflow.");
        result.trace.resize(trace_count);
        if (trace_count) check_cuda(cudaMemcpy(result.trace.data(), events.data, trace_count * sizeof(TraceEvent<T>), cudaMemcpyDeviceToHost));
    }
    result.download_seconds = seconds(download);
    result.total_seconds = seconds(start);
    return result;
}

template <typename T>
T replay_gpu(const std::vector<TraceEvent<T>>& events)
{
    if (events.empty()) return T{0};
    DeviceBuffer<TraceEvent<T>> de(events.size());
    DeviceBuffer<T> value(1);
    DeviceBuffer<int> error(1);
    upload(de, events);
    check_cuda(cudaMemset(value.data, 0, sizeof(T)));
    check_cuda(cudaMemset(error.data, 0, sizeof(int)));
    replay_events<T><<<1, 1>>>(de.data, events.size(), value.data, error.data);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    int mismatch = 0;
    T result = 0;
    check_cuda(cudaMemcpy(&mismatch, error.data, sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&result, value.data, sizeof(T), cudaMemcpyDeviceToHost));
    if (mismatch) throw std::runtime_error("GPU trace replay mismatch.");
    return result;
}

template <typename T>
std::vector<T> multiply_gpu_serial(const CSRMatrix& A, const CSRMatrix& P,
    const NumericInputs<T>& inputs, const CSRPattern& pattern)
{
    return execute_gpu(A, P, inputs, pattern, Strategy::serial, {}).values;
}

#define INSTANTIATE(T) \
    template ExecutionResult<T> execute_gpu<T>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<T>&, const CSRPattern&, Strategy, const ExecutionOptions&); \
    template std::vector<T> multiply_gpu_serial<T>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<T>&, const CSRPattern&); \
    template T replay_gpu<T>(const std::vector<TraceEvent<T>>&);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
}
