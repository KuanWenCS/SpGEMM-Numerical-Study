#include "matrix_analysis.hpp"
#include "matrix_io.hpp"

#include <limits>
#include <stdexcept>

namespace study {
namespace {

// Reject overflow instead of silently wrapping the counter.
Offset checked_add(Offset left, Offset right)
{
    if (right > std::numeric_limits<Offset>::max() - left) {
        throw std::overflow_error(
            "Structural product count exceeds 64-bit capacity.");
    }

    return left + right;
}

} // namespace

WorkloadStats analyze_workload(
    const CSRMatrix& A, const CSRMatrix& P)
{
    if (A.cols != P.rows) {
        throw std::invalid_argument(
            "A.cols must equal P.rows.");
    }

    validate_csr(A);
    validate_csr(P);

    WorkloadStats result;
    result.products_per_row.resize(A.rows, Offset{0});

    for (std::size_t i = 0; i < A.rows; ++i) {
        Offset row_products = 0;

        for (Offset p = A.row_ptr[i];
             p < A.row_ptr[i + 1]; ++p) {
            const std::size_t k = A.col_idx[p];

            const Offset p_row_length =
                P.row_ptr[k + 1] - P.row_ptr[k];

            row_products = checked_add(
                row_products, p_row_length);
        }

        result.products_per_row[i] = row_products;
        result.total_products = checked_add(
            result.total_products, row_products);

        if (row_products == 0) {
            ++result.zero_product_rows;
        }

        if (!result.max_row_index.has_value() ||
            row_products > result.max_products_per_row) {
            result.max_products_per_row = row_products;
            result.max_row_index = static_cast<Index>(i);
        }
    }

    if (A.rows != 0) {
        result.mean_products_per_row =
            static_cast<double>(result.total_products) /
            static_cast<double>(A.rows);
    }

    return result;
}

} // namespace study