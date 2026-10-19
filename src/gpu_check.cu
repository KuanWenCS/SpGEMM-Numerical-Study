#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>

#include "spgemm_common.hpp"

void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n",
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Store pointers to CSR arrays accessible by the GPU.
// This view does not own or release memory.
template <typename T>
struct CSRView {
    int rows;
    int cols;
    const int* row_ptr;
    const int* col_idx;
    const T* values;
};

// Copy one host vector into unified memory.
template <typename T>
T* copy_to_managed(const std::vector<T>& source)
{
    if (source.empty()) {
        return nullptr;
    }

    T* destination = nullptr;
    check(cudaMallocManaged(
        &destination, source.size() * sizeof(T)));

    for (std::size_t i = 0; i < source.size(); ++i) {
        destination[i] = source[i];
    }

    return destination;
}

// Assign one thread to each output row.
// Accumulate products in CSR storage order.
template <typename T>
__global__ void spgemm_rows(
    CSRView<T> A, CSRView<T> B, T* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= A.rows) {
        return;
    }

    const std::size_t row_start =
        static_cast<std::size_t>(i) * B.cols;

    for (int j = 0; j < B.cols; ++j) {
        output[row_start + j] = T{0};
    }

    for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        const int k = A.col_idx[p];

        for (int q = B.row_ptr[k]; q < B.row_ptr[k + 1]; ++q) {
            const int j = B.col_idx[q];
            const T product = A.values[p] * B.values[q];

            output[row_start + j] =
                output[row_start + j] + product;
        }
    }
}

template <typename T>
bool run_case()
{
    // Use the same input factory as the CPU baseline.
    const auto [A, B] = make_hand_case<T>();

    int* a_row_ptr = copy_to_managed(A.row_ptr);
    int* a_col_idx = copy_to_managed(A.col_idx);
    T* a_values = copy_to_managed(A.values);

    int* b_row_ptr = copy_to_managed(B.row_ptr);
    int* b_col_idx = copy_to_managed(B.col_idx);
    T* b_values = copy_to_managed(B.values);

    const CSRView<T> gpu_A{
        A.rows, A.cols, a_row_ptr, a_col_idx, a_values
    };

    const CSRView<T> gpu_B{
        B.rows, B.cols, b_row_ptr, b_col_idx, b_values
    };

    const std::size_t output_size =
        static_cast<std::size_t>(A.rows) * B.cols;

    T* output = nullptr;
    check(cudaMallocManaged(
        &output, output_size * sizeof(T)));

    constexpr int threads = 32;
    const int blocks = (A.rows + threads - 1) / threads;

    spgemm_rows<T><<<blocks, threads>>>(gpu_A, gpu_B, output);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    std::printf("%s\n", precision);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            const std::size_t index =
                static_cast<std::size_t>(i) * B.cols + j;

            std::printf("C(%d,%d) = %.17g\n",
                        i, j,
                        static_cast<double>(output[index]));
        }
    }

    const std::vector<T> expected{
        T{17}, T{16}, T{24}, T{53}
    };

    bool passed = output_size == expected.size();

    if (passed) {
        for (std::size_t p = 0; p < output_size; ++p) {
            if (output[p] != expected[p]) {
                passed = false;
            }
        }
    }

    std::printf("Check: %s\n\n", passed ? "PASS" : "FAIL");

    check(cudaFree(output));

    check(cudaFree(a_row_ptr));
    check(cudaFree(a_col_idx));
    check(cudaFree(a_values));

    check(cudaFree(b_row_ptr));
    check(cudaFree(b_col_idx));
    check(cudaFree(b_values));

    return passed;
}

int main()
{
    const bool fp32_passed = run_case<float>();
    const bool fp64_passed = run_case<double>();

    return fp32_passed && fp64_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}