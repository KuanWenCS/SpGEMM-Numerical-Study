#include <cstdio>
#include <type_traits>
#include <vector>
#include "spgemm_common.hpp"

// Compute C = A * B using a fixed, sequential traversal order.
template <typename T>
CSR<T> spgemm(const CSR<T>& A, const CSR<T>& B)
{
    CSR<T> C{A.rows, B.cols, {0}, {}, {}};

    for (int i = 0; i < A.rows; ++i) {
        std::vector<T> accumulator(B.cols, T{0});
        std::vector<bool> touched(B.cols, false);

        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            const int k = A.col_idx[p];

            for (int q = B.row_ptr[k]; q < B.row_ptr[k + 1]; ++q) {
                const int j = B.col_idx[q];

                const T product = A.values[p] * B.values[q];
                accumulator[j] = accumulator[j] + product;
                touched[j] = true;
            }
        }

        // Emit entries in increasing column order.
        // Keep structurally present entries even if their value is zero.
        for (int j = 0; j < B.cols; ++j) {
            if (touched[j]) {
                C.col_idx.push_back(j);
                C.values.push_back(accumulator[j]);
            }
        }

        C.row_ptr.push_back(static_cast<int>(C.values.size()));
    }

    return C;
}

template <typename T>
bool run_case()
{
    const auto [A, B] = make_hand_case<T>();

    const CSR<T> C = spgemm(A, B);
    const char* precision = std::is_same_v<T, float> ? "FP32" : "FP64";

    std::printf("%s\n", precision);
    for (int i = 0; i < C.rows; ++i) {
        for (int p = C.row_ptr[i]; p < C.row_ptr[i + 1]; ++p) {
            std::printf("C(%d,%d) = %.17g\n",
                        i, C.col_idx[p],
                        static_cast<double>(C.values[p]));
        }
    }

    // This small integer-valued case is exactly representable.
    const bool passed =
        C.row_ptr == std::vector<int>{0, 2, 4} &&
        C.col_idx == std::vector<int>{0, 1, 0, 1} &&
        C.values == std::vector<T>{T{17}, T{16}, T{24}, T{53}};

    std::printf("Check: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

int main()
{
    const bool fp32_passed = run_case<float>();
    const bool fp64_passed = run_case<double>();
    return fp32_passed && fp64_passed ? 0 : 1;
}