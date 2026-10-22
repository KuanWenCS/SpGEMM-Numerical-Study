#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "spgemm_gpu.hpp"

namespace {

template <typename T>
void run_precision(
    const study::CSRMatrix& A,
    const study::CSRMatrix& P,
    const study::CSRPattern& pattern)
{
    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    std::cerr << "Preparing shared " << precision << " inputs...\n";
    const auto inputs = study::prepare_numeric_inputs<T>(A, P);

    std::cerr << "Computing CPU " << precision << "...\n";
    const auto cpu = study::multiply_cpu_serial(
        A, P, inputs, pattern);

    std::cerr << "Computing GPU " << precision << "...\n";
    const auto gpu = study::multiply_gpu_serial(
        A, P, inputs, pattern);

    if (cpu.size() != gpu.size()) {
        throw std::runtime_error("Output lengths differ.");
    }

    study::Offset bitwise_differences = 0;
    study::Offset numeric_differences = 0;
    study::Offset nonfinite_pairs = 0;

    long double max_abs_difference = 0;

    for (std::size_t p = 0; p < cpu.size(); ++p) {
        if (std::memcmp(&cpu[p], &gpu[p], sizeof(T)) != 0) {
            ++bitwise_differences;
        }

        if (!std::isfinite(cpu[p]) || !std::isfinite(gpu[p])) {
            ++nonfinite_pairs;
            continue;
        }

        if (cpu[p] != gpu[p]) {
            ++numeric_differences;
        }

        const long double difference = std::fabs(
            static_cast<long double>(cpu[p]) -
            static_cast<long double>(gpu[p]));

        max_abs_difference = std::max(
            max_abs_difference, difference);
    }

    std::cout
        << precision << ','
        << cpu.size() << ','
        << bitwise_differences << ','
        << numeric_differences << ','
        << nonfinite_pairs << ','
        << std::setprecision(21) << max_abs_difference
        << '\n';

    if (bitwise_differences != 0 || nonfinite_pairs != 0) {
        std::cerr
            << precision
            << ": differences or non-finite values observed.\n";
    }
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
        int device = 0;
        cudaDeviceProp properties{};

        cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        std::cerr
            << "GPU: " << properties.name
            << ", compute capability "
            << properties.major << '.' << properties.minor
            << '\n';

        std::cerr << "Reading A...\n";
        const auto A = study::load_matrix_market(argv[1]);

        std::cerr << "Reading P...\n";
        const auto P = study::load_matrix_market(argv[2]);

        std::cerr << "Building output structure...\n";
        const auto symbolic = study::build_output_pattern(
            A.matrix, P.matrix);

        std::cout
            << "precision,output_entries,bitwise_differences,"
            << "numeric_differences,nonfinite_pairs,"
            << "max_abs_diff_finite_pairs\n";

        run_precision<float>(
            A.matrix, P.matrix, symbolic.pattern);

        run_precision<double>(
            A.matrix, P.matrix, symbolic.pattern);

        std::cerr << "CPU/GPU comparison completed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}