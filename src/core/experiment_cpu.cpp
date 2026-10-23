#include "experiment.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace study {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
T atomic_add(std::atomic<T>& destination, T product)
{
    T before = destination.load(std::memory_order_relaxed);
    for (;;) {
        const T after = before + product;
        if (destination.compare_exchange_weak(
                before, after, std::memory_order_relaxed)) {
            return before;
        }
    }
}
}

template <typename T>
ExecutionResult<T> execute_cpu(
    const CSRMatrix& A, const CSRMatrix& P, const NumericInputs<T>& inputs,
    const CSRPattern& pattern, Strategy strategy, const ExecutionOptions& options)
{
    const auto start = Clock::now();
    ExecutionResult<T> result;
    if (strategy == Strategy::serial) {
        if (options.trace_slot) throw std::invalid_argument("Trace requires atomic strategy.");
        result.values = multiply_cpu_serial(A, P, inputs, pattern);
        result.compute_seconds = seconds(start);
        result.total_seconds = result.compute_seconds;
        return result;
    }
    validate_csr(A);
    validate_csr(P);
    validate_output_pattern(A, P, pattern);
    if (inputs.a_values.size() != A.values.size() ||
        inputs.p_values.size() != P.values.size()) {
        throw std::invalid_argument("Typed input lengths differ.");
    }
    if (!options.threads || options.threads > 256) {
        throw std::invalid_argument("CPU threads must be in [1,256].");
    }
    if (options.trace_slot && (*options.trace_slot >= pattern.col_idx.size() ||
                              !options.trace_capacity)) {
        throw std::invalid_argument("Invalid trace selection.");
    }
    const auto size = pattern.col_idx.size();
    auto accumulator = std::make_unique<std::atomic<T>[]>(size);
    for (std::size_t p = 0; p < size; ++p) accumulator[p].store(T{0});
    result.values.resize(size);
    if (options.trace_slot) result.trace.resize(options.trace_capacity);
    std::mutex trace_mutex;
    Offset trace_count = 0;
    std::atomic<bool> error{false};
    std::vector<std::thread> workers;
    workers.reserve(options.threads);
    result.prepare_seconds = seconds(start);
    const auto compute_start = Clock::now();
    auto worker = [&](unsigned id) {
        std::size_t row = 0;
        for (Offset p = id; p < A.col_idx.size(); p += options.threads) {
            while (row < A.rows && A.row_ptr[row + 1] <= p) ++row;
            const Index k = A.col_idx[p];
            const auto begin = pattern.col_idx.begin() + pattern.row_ptr[row];
            const auto end = pattern.col_idx.begin() + pattern.row_ptr[row + 1];
            for (Offset q = P.row_ptr[k]; q < P.row_ptr[k + 1]; ++q) {
                const auto found = std::lower_bound(begin, end, P.col_idx[q]);
                if (found == end || *found != P.col_idx[q]) {
                    error.store(true);
                    continue;
                }
                const Offset slot = found - pattern.col_idx.begin();
                const T product = inputs.a_values[p] * inputs.p_values[q];
                if (options.trace_slot && slot == *options.trace_slot) {
                    std::lock_guard<std::mutex> lock(trace_mutex);
                    if (trace_count >= result.trace.size()) {
                        error.store(true);
                        continue;
                    }
                    const T before = atomic_add(accumulator[slot], product);
                    const T after = accumulator[slot].load(std::memory_order_relaxed);
                    result.trace[trace_count++] = {p, q, id, product, before, after};
                } else {
                    atomic_add(accumulator[slot], product);
                }
            }
        }
    };
    try {
        for (unsigned id = 0; id < options.threads; ++id) workers.emplace_back(worker, id);
    } catch (...) {
        for (auto& thread : workers) thread.join();
        throw;
    }
    for (auto& thread : workers) thread.join();
    result.compute_seconds = seconds(compute_start);
    if (error.load()) throw std::runtime_error("Atomic execution encountered invalid structure or trace capacity.");
    for (std::size_t p = 0; p < size; ++p) result.values[p] = accumulator[p].load();
    result.trace.resize(trace_count);
    result.total_seconds = seconds(start);
    return result;
}

template ExecutionResult<float> execute_cpu<float>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<float>&, const CSRPattern&, Strategy, const ExecutionOptions&);
template ExecutionResult<double> execute_cpu<double>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<double>&, const CSRPattern&, Strategy, const ExecutionOptions&);
}
