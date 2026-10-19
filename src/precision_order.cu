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

template <typename T>
struct TestData {
    T a[3];
    T b[3];
    T gpu_results[2];
};

// Compile this function for both CPU and GPU execution.
// This isolates the accumulation of one SpGEMM output entry.
template <typename T>
__host__ __device__
T accumulate_entry(const T* a, const T* b, bool reverse)
{
    T sum = T{0};

    for (int step = 0; step < 3; ++step) {
        const int k = reverse ? 2 - step : step;
        const T product = a[k] * b[k];
        sum = sum + product;
    }

    return sum;
}

// Thread 0 computes forward order; thread 1 computes reverse order.
template <typename T>
__global__ void evaluate_orders(TestData<T>* data)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 2) {
        return;
    }

    data->gpu_results[index] =
        accumulate_entry(data->a, data->b, index == 1);
}

template <typename T>
bool run_case()
{
    TestData<T>* data = nullptr;
    check(cudaMallocManaged(&data, sizeof(TestData<T>)));

    *data = TestData<T>{
        {T{33554432}, T{-33554432}, T{1}},
        {T{1}, T{1}, T{1}},
        {T{0}, T{0}}
    };

    const T cpu_results[2] = {
        accumulate_entry(data->a, data->b, false),
        accumulate_entry(data->a, data->b, true)
    };

    evaluate_orders<T><<<1, 2>>>(data);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    const char* precision =
        std::is_same_v<T, float> ? "FP32" : "FP64";

    const char* orders[2] = {"forward", "reverse"};
    bool passed = true;

    for (int index = 0; index < 2; ++index) {
        const T expected =
            std::is_same_v<T, float> && index == 1 ? T{0} : T{1};

        std::printf("%s,%s,CPU,%.17g,%.17g\n",
                    precision, orders[index],
                    static_cast<double>(cpu_results[index]),
                    static_cast<double>(cpu_results[index]) - 1.0);

        std::printf("%s,%s,GPU,%.17g,%.17g\n",
                    precision, orders[index],
                    static_cast<double>(data->gpu_results[index]),
                    static_cast<double>(data->gpu_results[index]) - 1.0);

        if (cpu_results[index] != expected ||
            data->gpu_results[index] != expected) {
            passed = false;
        }
    }

    check(cudaFree(data));
    return passed;
}

int main()
{
    std::printf("precision,order,device,value,signed_error\n");

    const bool fp32_passed = run_case<float>();
    const bool fp64_passed = run_case<double>();

    return fp32_passed && fp64_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}