#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <cuda_runtime.h>

void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n",
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Fixed CSR inputs and a dense output buffer for this small test.
template <typename T>
struct TestData {
    int a_row_ptr[3];
    int a_col_idx[4];
    T a_values[4];

    int b_row_ptr[4];
    int b_col_idx[4];
    T b_values[4];

    T output[4];
};

// Assign one thread to each output row.
// Each thread traverses its row in the same order as the CPU baseline.
template <typename T>
__global__ void spgemm_rows(TestData<T>* data, int rows, int cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows) {
        return;
    }

    for (int j = 0; j < cols; ++j) {
        data->output[i * cols + j] = T{0};
    }

    for (int p = data->a_row_ptr[i];
         p < data->a_row_ptr[i + 1]; ++p) {
        const int k = data->a_col_idx[p];

        for (int q = data->b_row_ptr[k];
             q < data->b_row_ptr[k + 1]; ++q) {
            const int j = data->b_col_idx[q];
            const T product = data->a_values[p] * data->b_values[q];

            data->output[i * cols + j] =
                data->output[i * cols + j] + product;
        }
    }
}

template <typename T>
bool run_case()
{
    constexpr int rows = 2;
    constexpr int cols = 2;

    TestData<T>* data = nullptr;
    check(cudaMallocManaged(&data, sizeof(TestData<T>)));

    // Initialize the same matrices used in the CPU baseline.
    *data = TestData<T>{
        {0, 2, 4},
        {0, 2, 1, 2},
        {T{1}, T{2}, T{3}, T{4}},

        {0, 1, 2, 4},
        {0, 1, 0, 1},
        {T{5}, T{7}, T{6}, T{8}},

        {T{0}, T{0}, T{0}, T{0}}
    };

    constexpr int threads = 32;
    constexpr int blocks = (rows + threads - 1) / threads;

    spgemm_rows<T><<<blocks, threads>>>(data, rows, cols);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    std::printf("%s\n", precision);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::printf("C(%d,%d) = %.17g\n",
                        i, j,
                        static_cast<double>(data->output[i * cols + j]));
        }
    }

    const T expected[4] = {T{17}, T{16}, T{24}, T{53}};
    bool passed = true;

    for (int p = 0; p < rows * cols; ++p) {
        if (data->output[p] != expected[p]) {
            passed = false;
        }
    }

    std::printf("Check: %s\n\n", passed ? "PASS" : "FAIL");

    check(cudaFree(data));
    return passed;
}

int main()
{
    const bool fp32_passed = run_case<float>();
    const bool fp64_passed = run_case<double>();

    return fp32_passed && fp64_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}