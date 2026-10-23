#ifndef STUDY_ERROR_ANALYSIS_HPP
#define STUDY_ERROR_ANALYSIS_HPP

#include <iosfwd>
#include <string>
#include "exact_binary.hpp"
#include "spgemm_numeric.hpp"

namespace study {

struct ReferenceEntry {
    Offset slot = 0;
    Index row = 0;
    Index column = 0;
    Offset products = 0;
    Offset a_row_entries = 0;
    Offset p_column_entries = 0;
    Offset c_row_entries = 0;
    Offset c_column_entries = 0;
    Offset positive_products = 0;
    Offset negative_products = 0;
    Offset zero_products = 0;
    ExactBinary value;
    ExactBinary rounded_sum;
    ExactBinary absolute_sum;
    ExactBinary min_nonzero_product;
    ExactBinary max_product;
    bool rounded_products_finite = true;
};

struct ErrorSummary {
    Offset samples = 0;
    Offset nonfinite = 0;
    Offset nonzero_errors = 0;
    Offset zero_references = 0;
    std::string max_absolute_error = "0";
    std::string max_relative_error = "NA";
    std::string worst_row = "NA";
    std::string worst_column = "NA";
};

template <typename T>
std::vector<ReferenceEntry> build_reference(
    const CSRMatrix&, const CSRMatrix&, const NumericInputs<T>&,
    const SymbolicResult&, Offset sample_limit);

void write_reference(std::ostream&, const std::vector<ReferenceEntry>&);
bool larger_range(double, double, double, double);
std::string range_text(double, double);

template <typename T>
ErrorSummary analyze_errors(
    const std::vector<ReferenceEntry>&, const std::vector<T>&,
    std::ostream*, const std::string& prefix);

}
#endif
