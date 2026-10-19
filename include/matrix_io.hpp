#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <istream>
#include <string>

#include "study_types.hpp"

namespace study {

// Read, normalize, and validate a real coordinate matrix.
MatrixInput read_matrix_market(std::istream& input);

// Open a plain-text Matrix Market file and read its contents.
MatrixInput load_matrix_market(const std::string& path);

// Check CSR structure, sorted columns, and finite values.
void validate_csr(const CSRMatrix& matrix);

} // namespace study

#endif