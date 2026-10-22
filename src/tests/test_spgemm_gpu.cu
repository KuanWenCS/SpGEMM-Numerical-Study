#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "spgemm_gpu.hpp"

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

} // namespace

int main()
{
    try {
        run_tests<float>();
        run_tests<double>();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}