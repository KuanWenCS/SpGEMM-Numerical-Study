#ifndef STUDY_EXPERIMENT_HPP
#define STUDY_EXPERIMENT_HPP

#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include "spgemm_numeric.hpp"

namespace study {

enum class Strategy { serial, atomic };

std::string gpu_description();

struct ExecutionOptions {
    unsigned threads = 8;
    unsigned gpu_threads = 128;
    std::optional<Offset> trace_slot;
    Offset trace_capacity = 0;
};

template <typename T>
struct TraceEvent {
    Offset a_position = 0;
    Offset p_position = 0;
    Offset worker = 0;
    T product = 0;
    T before = 0;
    T after = 0;
};

template <typename T>
struct ExecutionResult {
    std::vector<T> values;
    std::vector<TraceEvent<T>> trace;
    double prepare_seconds = 0;
    double compute_seconds = 0;
    double download_seconds = 0;
    double total_seconds = 0;
};

template <typename T>
std::uint64_t value_bits(T value)
{
    std::uint64_t bits = 0;
    if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
        std::uint32_t small = 0;
        std::memcpy(&small, &value, sizeof(T));
        bits = small;
    } else {
        std::memcpy(&bits, &value, sizeof(T));
    }
    return bits;
}

template <typename T>
ExecutionResult<T> execute_cpu(
    const CSRMatrix&, const CSRMatrix&, const NumericInputs<T>&,
    const CSRPattern&, Strategy, const ExecutionOptions&);

template <typename T>
ExecutionResult<T> execute_gpu(
    const CSRMatrix&, const CSRMatrix&, const NumericInputs<T>&,
    const CSRPattern&, Strategy, const ExecutionOptions&);

template <typename T>
T replay_cpu(const std::vector<TraceEvent<T>>& events)
{
    T sum = T{0};
    for (const auto& event : events) {
        if (value_bits(sum) != value_bits(event.before)) {
            throw std::runtime_error("Trace predecessor mismatch.");
        }
        sum = sum + event.product;
        if (value_bits(sum) != value_bits(event.after)) {
            throw std::runtime_error("Trace successor mismatch.");
        }
    }
    return sum;
}

template <typename T>
T replay_gpu(const std::vector<TraceEvent<T>>&);

}
#endif
