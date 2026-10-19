#ifndef STUDY_TYPES_HPP
#define STUDY_TYPES_HPP

#include <cstdint>
#include <vector>

namespace study {

using Index = std::uint32_t;
using Offset = std::uint64_t;

// Own the CSR arrays on the CPU.
struct CSRMatrix {
    Index rows = 0;
    Index cols = 0;

    std::vector<Offset> row_ptr;
    std::vector<Index> col_idx;
    std::vector<double> values;
};

// Keep source-format information separate from the CSR arrays.
struct MatrixMarketInfo {
    Offset stored_entries = 0;
    bool symmetric = false;
};

struct MatrixInput {
    CSRMatrix matrix;
    MatrixMarketInfo source;
};

} // namespace study

#endif