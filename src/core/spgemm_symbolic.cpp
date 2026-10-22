#include "spgemm_symbolic.hpp"
#include "matrix_io.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace study {
namespace {

Offset checked_add(Offset left, Offset right)
{
    if (right > std::numeric_limits<Offset>::max() - left) {
        throw std::overflow_error(
            "Structural product count exceeds 64-bit capacity.");
    }

    return left + right;
}

} // namespace

SymbolicResult build_output_pattern(
    const CSRMatrix& A, const CSRMatrix& P)
{
    if (A.cols != P.rows) {
        throw std::invalid_argument(
            "A.cols must equal P.rows.");
    }

    validate_csr(A);
    validate_csr(P);

    SymbolicResult result;
    result.pattern.rows = A.rows;
    result.pattern.cols = P.cols;

    result.pattern.row_ptr.reserve(
        static_cast<std::size_t>(A.rows) + 1);
    result.pattern.row_ptr.push_back(0);

    // Reuse one workspace across all output rows.
    std::vector<Offset> counts(P.cols, Offset{0});
    std::vector<Index> touched_columns;

    for (std::size_t i = 0; i < A.rows; ++i) {
        touched_columns.clear();

        for (Offset p = A.row_ptr[i];
             p < A.row_ptr[i + 1]; ++p) {
            const std::size_t k = A.col_idx[p];

            for (Offset q = P.row_ptr[k];
                 q < P.row_ptr[k + 1]; ++q) {
                const Index j = P.col_idx[q];

                if (counts[j] == 0) {
                    touched_columns.push_back(j);
                }

                counts[j] = checked_add(counts[j], 1);
            }
        }

        // Fix output storage order without changing product traversal.
        std::sort(
            touched_columns.begin(), touched_columns.end());

        for (const Index j : touched_columns) {
            result.pattern.col_idx.push_back(j);
            result.products_per_entry.push_back(counts[j]);

            result.total_products = checked_add(
                result.total_products, counts[j]);

            // Clear only the columns touched by this row.
            counts[j] = 0;
        }

        result.pattern.row_ptr.push_back(
            static_cast<Offset>(
                result.pattern.col_idx.size()));
    }

    return result;
}

} // namespace study