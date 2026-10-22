#ifndef MATRIX_ANALYSIS_HPP
#define MATRIX_ANALYSIS_HPP

#include <optional>
#include <vector>

#include "study_types.hpp"

namespace study {

struct WorkloadStats {
    std::vector<Offset> products_per_row;

    Offset total_products = 0;
    Offset max_products_per_row = 0;
    Offset zero_product_rows = 0;

    double mean_products_per_row = 0;

    // The first row attaining the maximum; absent if A has no rows.
    std::optional<Index> max_row_index;
};

// Validate the inputs and count structural products for C = A * P.
// Explicit zeros are retained and their paths are counted.
WorkloadStats analyze_workload(
    const CSRMatrix& A, const CSRMatrix& P);

} // namespace study

#endif