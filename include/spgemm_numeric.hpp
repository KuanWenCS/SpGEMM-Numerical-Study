#ifndef SPGEMM_NUMERIC_HPP
#define SPGEMM_NUMERIC_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "matrix_io.hpp"
#include "spgemm_symbolic.hpp"

namespace study {

// Own precision-specific values without copying CSR structure.
template <typename T>
struct NumericInputs {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, double>,
        "Only float and double are supported.");

    std::vector<T> a_values;
    std::vector<T> p_values;
};

// Convert once before numerical computation.
template <typename T>
std::vector<T> convert_values(const std::vector<double>& source)
{
    std::vector<T> result;
    result.reserve(source.size());

    const double limit =
        static_cast<double>(std::numeric_limits<T>::max());

    for (const double value : source) {
        if (!std::isfinite(value) || std::fabs(value) > limit) {
            throw std::runtime_error(
                "Input value is not finite or exceeds target range.");
        }

        result.push_back(static_cast<T>(value));
    }

    return result;
}

template <typename T>
NumericInputs<T> prepare_numeric_inputs(
    const CSRMatrix& A, const CSRMatrix& P)
{
    validate_csr(A);
    validate_csr(P);

    return {
        convert_values<T>(A.values),
        convert_values<T>(P.values)
    };
}

// Validate output layout without allocating numerical values.
inline void validate_output_pattern(
    const CSRMatrix& A,
    const CSRMatrix& P,
    const CSRPattern& pattern)
{
    if (A.cols != P.rows ||
        pattern.rows != A.rows ||
        pattern.cols != P.cols) {
        throw std::invalid_argument(
            "Incompatible input or output dimensions.");
    }

    if (pattern.row_ptr.size() !=
            static_cast<std::size_t>(pattern.rows) + 1 ||
        pattern.row_ptr.front() != 0 ||
        pattern.row_ptr.back() != pattern.col_idx.size()) {
        throw std::runtime_error(
            "Invalid output row offsets.");
    }

    for (std::size_t i = 0; i < pattern.rows; ++i) {
        const Offset begin = pattern.row_ptr[i];
        const Offset end = pattern.row_ptr[i + 1];

        if (begin > end || end > pattern.col_idx.size()) {
            throw std::runtime_error(
                "Invalid output row interval.");
        }

        for (Offset p = begin; p < end; ++p) {
            if (pattern.col_idx[p] >= pattern.cols ||
                (p > begin &&
                 pattern.col_idx[p - 1] >= pattern.col_idx[p])) {
                throw std::runtime_error(
                    "Invalid or unsorted output columns.");
            }
        }
    }
}

// Compute values aligned with the supplied output pattern.
// Products are accumulated in A-then-P CSR traversal order.
template <typename T>
std::vector<T> multiply_cpu_serial(
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

    // Reuse one accumulator across all output rows.
    std::vector<T> accumulator(P.cols, T{0});
    std::vector<unsigned char> touched(P.cols, 0);

    for (std::size_t i = 0; i < A.rows; ++i) {
        Offset touched_count = 0;

        for (Offset p = A.row_ptr[i];
             p < A.row_ptr[i + 1]; ++p) {
            const std::size_t k = A.col_idx[p];

            for (Offset q = P.row_ptr[k];
                 q < P.row_ptr[k + 1]; ++q) {
                const Index j = P.col_idx[q];

                if (!touched[j]) {
                    touched[j] = 1;
                    accumulator[j] = T{0};
                    ++touched_count;
                }

                const T product =
                    inputs.a_values[p] * inputs.p_values[q];

                accumulator[j] = accumulator[j] + product;
            }
        }

        const Offset begin = pattern.row_ptr[i];
        const Offset end = pattern.row_ptr[i + 1];

        if (touched_count != end - begin) {
            throw std::runtime_error(
                "Output pattern does not match product paths.");
        }

        for (Offset p = begin; p < end; ++p) {
            const Index j = pattern.col_idx[p];

            if (!touched[j]) {
                throw std::runtime_error(
                    "Output pattern contains an unexpected column.");
            }

            output[p] = accumulator[j];
            touched[j] = 0;
        }
    }

    return output;
}

} // namespace study

#endif