#include <cstdio>
#include <cstring>
#include <limits>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "spgemm_gpu.hpp"
#include "experiment.hpp"

namespace {

template <typename T>
void check_case(
    const char* name,
    const study::CSRMatrix& A,
    const study::CSRMatrix& P,
    const std::vector<T>& expected)
{
    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    const auto cpu = study::multiply_cpu_serial(
        A, P, inputs, symbolic.pattern);

    const auto gpu = study::multiply_gpu_serial(
        A, P, inputs, symbolic.pattern);

    if (cpu != expected || gpu != expected) {
        throw std::runtime_error(
            "CPU or GPU differs from the known answer.");
    }

    for (std::size_t p = 0; p < cpu.size(); ++p) {
        if (std::memcmp(&cpu[p], &gpu[p], sizeof(T)) != 0) {
            throw std::runtime_error(
                "CPU/GPU bitwise mismatch in controlled case.");
        }
    }

    std::printf(
        "%s %s: PASS\n",
        std::is_same_v<T, float> ? "FP32" : "FP64",
        name);
}

template <typename T>
void run_tests()
{
    const study::CSRMatrix A{
        2, 3,
        {0, 2, 4},
        {0, 2, 1, 2},
        {1, 2, 3, 4}
    };

    const study::CSRMatrix P{
        3, 2,
        {0, 1, 2, 4},
        {0, 1, 0, 1},
        {5, 7, 6, 8}
    };

    check_case<T>("hand_case", A, P, {17, 16, 24, 53});

    const study::CSRMatrix order_A{
        1, 3,
        {0, 3},
        {0, 1, 2},
        {33554432, 1, -33554432}
    };

    const study::CSRMatrix order_P{
        3, 1,
        {0, 1, 2, 3},
        {0, 0, 0},
        {1, 1, 1}
    };

    const T expected =
        std::is_same_v<T, float> ? T{0} : T{1};

    check_case<T>(
        "precision_effect", order_A, order_P, {expected});

    const study::CSRMatrix zero_A{
        1, 3,
        {0, 3},
        {0, 1, 2},
        {1, -1, 0}
    };

    const study::CSRMatrix zero_P{
        3, 2,
        {0, 1, 2, 3},
        {0, 0, 1},
        {1, 1, 5}
    };

    check_case<T>(
        "zero_and_cancellation", zero_A, zero_P, {0, 0});

    const study::CSRMatrix empty_A{
        2, 1, {0, 0, 0}, {}, {}
    };

    const study::CSRMatrix empty_P{
        1, 1, {0, 1}, {0}, {1}
    };

    check_case<T>(
        "empty_output", empty_A, empty_P, {});
}

template <typename T>
void test_parallel()
{
    study::CSRMatrix A;
    A.rows = 1; A.cols = 256; A.row_ptr = {0, 256};
    study::CSRMatrix P;
    P.rows = 256; P.cols = 1;
    for (study::Index k = 0; k < 256; ++k) {
        A.col_idx.push_back(k); A.values.push_back(1);
        P.row_ptr.push_back(k); P.col_idx.push_back(0); P.values.push_back(1);
    }
    P.row_ptr.push_back(256);
    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);
    study::ExecutionOptions options;
    for (const unsigned threads : {32U, 128U, 256U}) {
        options.gpu_threads = threads;
        const auto native = study::execute_gpu(A, P, inputs, symbolic.pattern, study::Strategy::atomic, options);
        if (native.values != std::vector<T>{256}) throw std::runtime_error("GPU atomic updates were lost.");
        options.trace_slot = 0;
        options.trace_capacity = 256;
        const auto diagnostic = study::execute_gpu(A, P, inputs, symbolic.pattern, study::Strategy::atomic, options);
        if (diagnostic.trace.size() != 256 ||
            study::value_bits(study::replay_gpu(diagnostic.trace)) != study::value_bits(diagnostic.values[0])) {
            throw std::runtime_error("GPU trace replay failed.");
        }
        options.trace_slot.reset();
    }
    const T tiny = std::numeric_limits<T>::denorm_min();
    const study::CSRMatrix tiny_A{1, 1, {0, 1}, {0}, {static_cast<double>(tiny)}};
    const study::CSRMatrix tiny_P{1, 1, {0, 1}, {0}, {1}};
    const auto ts = study::build_output_pattern(tiny_A, tiny_P);
    const auto ti = study::prepare_numeric_inputs<T>(tiny_A, tiny_P);
    const auto serial = study::execute_gpu(tiny_A, tiny_P, ti, ts.pattern, study::Strategy::serial, options);
    if (study::value_bits(serial.values[0]) != study::value_bits(tiny)) throw std::runtime_error("Serial GPU subnormal was flushed.");
    options.trace_slot = 0;
    options.trace_capacity = 1;
    const auto atomic = study::execute_gpu(tiny_A, tiny_P, ti, ts.pattern, study::Strategy::atomic, options);
    if (atomic.trace.size() != 1 || study::value_bits(study::replay_gpu(atomic.trace)) != study::value_bits(atomic.values[0])) {
        throw std::runtime_error("Subnormal atomic trace replay failed.");
    }
    std::printf("%s subnormal_probe serial_bits=%llu atomic_bits=%llu\n",
        std::is_same_v<T, float> ? "FP32" : "FP64",
        static_cast<unsigned long long>(study::value_bits(serial.values[0])),
        static_cast<unsigned long long>(study::value_bits(atomic.values[0])));
    std::printf("%s atomic_and_trace: PASS\n", std::is_same_v<T, float> ? "FP32" : "FP64");
}

}

int main()
{
    try {
        run_tests<float>();
        run_tests<double>();
        test_parallel<float>();
        test_parallel<double>();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
