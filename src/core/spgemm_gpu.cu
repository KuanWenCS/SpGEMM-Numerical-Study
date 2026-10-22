#include "spgemm_gpu.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace study {
namespace {

void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") +
            cudaGetErrorString(status));
    }
}

// Own one device allocation and release it on scope exit.
template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(std::size_t count)
    {
        if (count == 0) {
            return;
        }

        if (count >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error(
                "Device allocation size overflow.");
        }

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&data),
            count * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
void upload(
    const DeviceBuffer<T>& destination,
    const std::vector<T>& source)
{
    if (!source.empty()) {
        check_cuda(cudaMemcpy(
            destination.data,
            source.data(),
            source.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }
}

template <typename T>
struct DeviceCSRView {
    Index rows;
    const Offset* row_ptr;
    const Index* col_idx;
    const T* values;
};

// Return the first position whose column is not less than target.
__device__
Offset find_column(
    const Index* columns,
    Offset begin,
    Offset end,
    Index target)
{
    while (begin < end) {
        const Offset middle = begin + (end - begin) / 2;

        if (columns[middle] < target) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }

    return begin;
}

// Each thread processes complete rows sequentially.
template <typename T>
__global__ void multiply_rows(
    DeviceCSRView<T> A,
    DeviceCSRView<T> P,
    const Offset* c_row_ptr,
    const Index* c_col_idx,
    T* output,
    int* error)
{
    const Offset first_row =
        static_cast<Offset>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const Offset stride =
        static_cast<Offset>(gridDim.x) * blockDim.x;

    for (Offset i = first_row; i < A.rows; i += stride) {
        const Offset row_begin = c_row_ptr[i];
        const Offset row_end = c_row_ptr[i + 1];

        for (Offset slot = row_begin; slot < row_end; ++slot) {
            output[slot] = T{0};
        }

        for (Offset p = A.row_ptr[i];
             p < A.row_ptr[i + 1]; ++p) {
            const Index k = A.col_idx[p];

            for (Offset q = P.row_ptr[k];
                 q < P.row_ptr[k + 1]; ++q) {
                const Index j = P.col_idx[q];

                const Offset slot = find_column(
                    c_col_idx, row_begin, row_end, j);

                if (slot == row_end || c_col_idx[slot] != j) {
                    atomicExch(error, 1);
                    return;
                }

                const T product = A.values[p] * P.values[q];
                output[slot] = output[slot] + product;
            }
        }
    }
}

} // namespace

template <typename T>
std::vector<T> multiply_gpu_serial(
    const CSRMatrix& A,
    const CSRMatrix& P,
    const NumericInputs<T>& inputs,
    const CSRPattern& pattern)
{
    validate_csr(A);
    validate_csr(P);
    validate_output_pattern(A, P, pattern);

    if (inputs.a_values.size() != A.values.size() ||
        inputs.p_values.size() != P.values.size()) {
        throw std::invalid_argument(
            "Precision-specific input lengths do not match CSR.");
    }

    std::vector<T> output(pattern.col_idx.size(), T{0});

    if (output.empty()) {
        return output;
    }

    DeviceBuffer<Offset> a_row_ptr(A.row_ptr.size());
    DeviceBuffer<Index> a_col_idx(A.col_idx.size());
    DeviceBuffer<T> a_values(inputs.a_values.size());

    DeviceBuffer<Offset> p_row_ptr(P.row_ptr.size());
    DeviceBuffer<Index> p_col_idx(P.col_idx.size());
    DeviceBuffer<T> p_values(inputs.p_values.size());

    DeviceBuffer<Offset> c_row_ptr(pattern.row_ptr.size());
    DeviceBuffer<Index> c_col_idx(pattern.col_idx.size());
    DeviceBuffer<T> c_values(output.size());

    DeviceBuffer<int> error(1);

    upload(a_row_ptr, A.row_ptr);
    upload(a_col_idx, A.col_idx);
    upload(a_values, inputs.a_values);

    upload(p_row_ptr, P.row_ptr);
    upload(p_col_idx, P.col_idx);
    upload(p_values, inputs.p_values);

    upload(c_row_ptr, pattern.row_ptr);
    upload(c_col_idx, pattern.col_idx);

    check_cuda(cudaMemset(error.data, 0, sizeof(int)));

    const DeviceCSRView<T> gpu_A{
        A.rows, a_row_ptr.data, a_col_idx.data, a_values.data
    };

    const DeviceCSRView<T> gpu_P{
        P.rows, p_row_ptr.data, p_col_idx.data, p_values.data
    };

    constexpr unsigned int threads = 128;

    const Offset required_blocks =
        (static_cast<Offset>(A.rows) + threads - 1) / threads;

    const unsigned int blocks = static_cast<unsigned int>(
        std::min<Offset>(required_blocks, 65535));

    multiply_rows<T><<<blocks, threads>>>(
        gpu_A,
        gpu_P,
        c_row_ptr.data,
        c_col_idx.data,
        c_values.data,
        error.data);

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    int host_error = 0;
    check_cuda(cudaMemcpy(
        &host_error,
        error.data,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    if (host_error != 0) {
        throw std::runtime_error(
            "GPU encountered a column missing from the output pattern.");
    }

    check_cuda(cudaMemcpy(
        output.data(),
        c_values.data,
        output.size() * sizeof(T),
        cudaMemcpyDeviceToHost));

    return output;
}

// Generate the two supported implementations in this translation unit.
template std::vector<float> multiply_gpu_serial<float>(
    const CSRMatrix&,
    const CSRMatrix&,
    const NumericInputs<float>&,
    const CSRPattern&);

template std::vector<double> multiply_gpu_serial<double>(
    const CSRMatrix&,
    const CSRMatrix&,
    const NumericInputs<double>&,
    const CSRPattern&);

} // namespace study