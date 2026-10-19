#ifndef SPGEMM_COMMON_HPP
#define SPGEMM_COMMON_HPP

#include <utility>
#include <vector>

// CSR indices are zero-based.
// Storage order is preserved and defines the baseline traversal order.
template <typename T>
struct CSR {
    int rows;
    int cols;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;
};

// Return the small, exactly representable validation inputs.
template <typename T>
std::pair<CSR<T>, CSR<T>> make_hand_case()
{
    CSR<T> A{
        2, 3,
        {0, 2, 4},
        {0, 2, 1, 2},
        {T{1}, T{2}, T{3}, T{4}}
    };

    CSR<T> B{
        3, 2,
        {0, 1, 2, 4},
        {0, 1, 0, 1},
        {T{5}, T{7}, T{6}, T{8}}
    };

    return {std::move(A), std::move(B)};
}

#endif