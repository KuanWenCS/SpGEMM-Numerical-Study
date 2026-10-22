#ifndef SPGEMM_GPU_HPP
#define SPGEMM_GPU_HPP

#include <vector>

#include "spgemm_numeric.hpp"

namespace study {

// Compute values aligned with the supplied sparse output pattern.
// Input values are copied without further precision conversion.
template <typename T>
std::vector<T> multiply_gpu_serial(
    const CSRMatrix& A,
    const CSRMatrix& P,
    const NumericInputs<T>& inputs,
    const CSRPattern& pattern);

} // namespace study

#endif