#ifndef SPGEMM_SYMBOLIC_HPP
#define SPGEMM_SYMBOLIC_HPP

#include <vector>

#include "study_types.hpp"

namespace study {

// Store output coordinates without numerical values.
struct CSRPattern {
    Index rows = 0;
    Index cols = 0;

    std::vector<Offset> row_ptr;
    std::vector<Index> col_idx;
};

struct SymbolicResult {
    CSRPattern pattern;

    // Aligned with pattern.col_idx.
    std::vector<Offset> products_per_entry;

    Offset total_products = 0;
};

// Find the structural output of C = A * P.
// Count paths even when an input value is explicitly zero.
SymbolicResult build_output_pattern(
    const CSRMatrix& A, const CSRMatrix& P);

} // namespace study

#endif