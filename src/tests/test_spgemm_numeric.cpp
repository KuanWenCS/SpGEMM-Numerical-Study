#include <cstdio>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "spgemm_numeric.hpp"

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void test_hand_case()
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

    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    const auto output = study::multiply_cpu_serial(
        A, P, inputs, symbolic.pattern);

    require(
        output == std::vector<T>{17, 16, 24, 53},
        "Incorrect hand-case values.");
}

template <typename T>
void test_zero_and_cancellation()
{
    const study::CSRMatrix A{
        1, 3,
        {0, 3},
        {0, 1, 2},
        {1, -1, 0}
    };

    const study::CSRMatrix P{
        3, 2,
        {0, 1, 2, 3},
        {0, 0, 1},
        {1, 1, 5}
    };

    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    const auto output = study::multiply_cpu_serial(
        A, P, inputs, symbolic.pattern);

    require(
        symbolic.pattern.col_idx ==
            std::vector<study::Index>{0, 1},
        "Zero-valued structural entries were lost.");

    require(
        output == std::vector<T>{0, 0},
        "Incorrect zero or cancellation result.");
}

template <typename T>
void test_precision_effect()
{
    const study::CSRMatrix A{
        1, 3,
        {0, 3},
        {0, 1, 2},
        {33554432, 1, -33554432}
    };

    const study::CSRMatrix P{
        3, 1,
        {0, 1, 2, 3},
        {0, 0, 0},
        {1, 1, 1}
    };

    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    const auto output = study::multiply_cpu_serial(
        A, P, inputs, symbolic.pattern);

    const T expected =
        std::is_same_v<T, float> ? T{0} : T{1};

    require(
        output.size() == 1 && output[0] == expected,
        "Unexpected precision-dependent result.");
}

template <typename T>
void test_empty_output()
{
    const study::CSRMatrix A{
        2, 1, {0, 0, 0}, {}, {}
    };

    const study::CSRMatrix P{
        1, 1, {0, 1}, {0}, {1}
    };

    const auto symbolic = study::build_output_pattern(A, P);
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    const auto output = study::multiply_cpu_serial(
        A, P, inputs, symbolic.pattern);

    require(output.empty(), "Expected empty output.");
}

template <typename T>
void run_tests()
{
    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    test_hand_case<T>();
    std::printf("%s hand_case: PASS\n", precision);

    test_zero_and_cancellation<T>();
    std::printf("%s zero_and_cancellation: PASS\n", precision);

    test_precision_effect<T>();
    std::printf("%s precision_effect: PASS\n", precision);

    test_empty_output<T>();
    std::printf("%s empty_output: PASS\n", precision);
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