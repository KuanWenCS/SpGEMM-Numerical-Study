#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <type_traits>

#include "spgemm_numeric.hpp"

namespace {

template <typename T>
bool run_precision(
    const study::CSRMatrix& A,
    const study::CSRMatrix& P,
    const study::CSRPattern& pattern)
{
    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    std::cerr << "Preparing " << precision << " inputs...\n";
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    std::cerr << "Computing CPU " << precision << "...\n";
    const auto output = study::multiply_cpu_serial(
        A, P, inputs, pattern);

    study::Offset zero_entries = 0;
    study::Offset nonfinite_entries = 0;
    double max_abs_finite = 0;

    for (const T value : output) {
        if (!std::isfinite(value)) {
            ++nonfinite_entries;
            continue;
        }

        if (value == T{0}) {
            ++zero_entries;
        }

        max_abs_finite = std::max(
            max_abs_finite,
            std::fabs(static_cast<double>(value)));
    }

    std::cout
        << precision << ','
        << output.size() << ','
        << zero_entries << ','
        << nonfinite_entries << ','
        << std::setprecision(17) << max_abs_finite << '\n';

    return nonfinite_entries == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0] << " A.mtx P.mtx\n";
        return 1;
    }

    try {
        std::cerr << "Reading A...\n";
        const auto A = study::load_matrix_market(argv[1]);

        std::cerr << "Reading P...\n";
        const auto P = study::load_matrix_market(argv[2]);

        std::cerr << "Building output structure...\n";
        const auto symbolic = study::build_output_pattern(
            A.matrix, P.matrix);

        std::cout
            << "precision,output_entries,zero_entries,"
            << "nonfinite_entries,max_abs_finite\n";

        const bool fp32_finite = run_precision<float>(
            A.matrix, P.matrix, symbolic.pattern);

        const bool fp64_finite = run_precision<double>(
            A.matrix, P.matrix, symbolic.pattern);

        std::cerr << "CPU numerical computation completed.\n";

        if (!fp32_finite || !fp64_finite) {
            std::cerr
                << "WARNING: non-finite output values detected.\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}