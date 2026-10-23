#include "error_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "experiment.hpp"

namespace study {
namespace {
using Decimal = boost::multiprecision::cpp_dec_float_100;

ExactBinary difference(ExactBinary left, ExactBinary right)
{
    right.significand = -right.significand;
    accumulate_exact(left, right);
    return left;
}

ExactBinary magnitude(ExactBinary value)
{
    if (value.significand < 0) value.significand = -value.significand;
    return value;
}

bool greater(const ExactBinary& left, const ExactBinary& right)
{
    return difference(left, right).significand > 0;
}

ExactBinary product(const ExactBinary& a, const ExactBinary& b)
{
    return {ExactInteger(a.significand * b.significand), a.exponent + b.exponent};
}

Decimal decimal(const ExactBinary& value)
{
    Decimal result(value.significand.str());
    return boost::multiprecision::ldexp(result, value.exponent);
}

std::string text(const Decimal& value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(24) << value;
    return out.str();
}

std::vector<Offset> sample_slots(Offset entries, Offset limit)
{
    const Offset count = std::min(entries, limit);
    std::vector<Offset> slots;
    slots.reserve(count);
    for (Offset i = 0; i < count; ++i) {
        if (count == entries || count < 2) slots.push_back(i);
        else {
            const Offset quotient = (entries - 1) / (count - 1);
            const Offset remainder = (entries - 1) % (count - 1);
            if (remainder && i > std::numeric_limits<Offset>::max() / remainder) {
                throw std::overflow_error("Sample index overflow.");
            }
            slots.push_back(i * quotient + i * remainder / (count - 1));
        }
    }
    return slots;
}
}

bool larger_range(double low, double high, double old_low, double old_high)
{
    return greater(difference(exact_from_double(high), exact_from_double(low)),
                   difference(exact_from_double(old_high), exact_from_double(old_low)));
}

std::string range_text(double low, double high)
{
    return text(decimal(difference(exact_from_double(high), exact_from_double(low))));
}

template <typename T>
std::vector<ReferenceEntry> build_reference(
    const CSRMatrix& A, const CSRMatrix& P, const NumericInputs<T>& inputs,
    const SymbolicResult& symbolic, Offset sample_limit)
{
    validate_csr(A);
    validate_csr(P);
    validate_output_pattern(A, P, symbolic.pattern);
    const auto& pattern = symbolic.pattern;
    if (inputs.a_values.size() != A.values.size() || inputs.p_values.size() != P.values.size() ||
        symbolic.products_per_entry.size() != pattern.col_idx.size()) {
        throw std::invalid_argument("Reference input lengths differ.");
    }
    const auto slots = sample_slots(pattern.col_idx.size(), sample_limit);
    std::vector<ReferenceEntry> result(slots.size());
    if (slots.empty()) return result;
    std::vector<Offset> p_columns(P.cols, 0), c_columns(P.cols, 0);
    for (auto j : P.col_idx) ++p_columns[j];
    for (auto j : pattern.col_idx) ++c_columns[j];
    const Offset absent = std::numeric_limits<Offset>::max();
    std::vector<Offset> selected(P.cols, absent);
    std::size_t row = 0;
    for (std::size_t r = 0; r < slots.size(); ++r) {
        while (row < A.rows && pattern.row_ptr[row + 1] <= slots[r]) ++row;
        auto& entry = result[r];
        entry.slot = slots[r];
        entry.row = static_cast<Index>(row);
        entry.column = pattern.col_idx[slots[r]];
        entry.a_row_entries = A.row_ptr[row + 1] - A.row_ptr[row];
        entry.p_column_entries = p_columns[entry.column];
        entry.c_row_entries = pattern.row_ptr[row + 1] - pattern.row_ptr[row];
        entry.c_column_entries = c_columns[entry.column];
    }
    for (std::size_t first = 0; first < result.size();) {
        std::size_t last = first + 1;
        const Index i = result[first].row;
        while (last < result.size() && result[last].row == i) ++last;
        for (auto r = first; r < last; ++r) selected[result[r].column] = r;
        for (Offset p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            const Index k = A.col_idx[p];
            for (Offset q = P.row_ptr[k]; q < P.row_ptr[k + 1]; ++q) {
                const Offset r = selected[P.col_idx[q]];
                if (r == absent) continue;
                auto& entry = result[r];
                const auto exact_product = multiply_exact(
                    decode_binary(static_cast<double>(inputs.a_values[p])),
                    decode_binary(static_cast<double>(inputs.p_values[q])));
                accumulate_exact(entry.value, exact_product);
                const auto absolute_product = magnitude(exact_product);
                accumulate_exact(entry.absolute_sum, absolute_product);
                if (exact_product.significand > 0) ++entry.positive_products;
                else if (exact_product.significand < 0) ++entry.negative_products;
                else ++entry.zero_products;
                if (absolute_product.significand != 0 &&
                    (entry.min_nonzero_product.significand == 0 ||
                     greater(entry.min_nonzero_product, absolute_product))) {
                    entry.min_nonzero_product = absolute_product;
                }
                if (greater(absolute_product, entry.max_product)) entry.max_product = absolute_product;
                const T rounded = inputs.a_values[p] * inputs.p_values[q];
                if (std::isfinite(rounded)) {
                    accumulate_exact(entry.rounded_sum, exact_from_double(static_cast<double>(rounded)));
                } else entry.rounded_products_finite = false;
                ++entry.products;
            }
        }
        for (auto r = first; r < last; ++r) {
            if (result[r].products != symbolic.products_per_entry[result[r].slot]) {
                throw std::runtime_error("Reference product count mismatch.");
            }
            selected[result[r].column] = absent;
        }
        first = last;
    }
    return result;
}

void write_reference(std::ostream& out, const std::vector<ReferenceEntry>& entries)
{
    out << "slot,row,column,products,a_row_entries,p_column_entries,c_row_entries,c_column_entries,"
           "positive_products,negative_products,zero_products,reference_significand,reference_exponent,"
           "reference_approx,rounded_sum_significand,rounded_sum_exponent,rounded_products_finite,"
           "sum_abs_products,min_nonzero_abs_product,max_abs_product,cancellation_condition\n";
    for (const auto& e : entries) {
        out << e.slot << ',' << e.row << ',' << e.column << ',' << e.products << ','
            << e.a_row_entries << ',' << e.p_column_entries << ',' << e.c_row_entries << ','
            << e.c_column_entries << ',' << e.positive_products << ',' << e.negative_products << ','
            << e.zero_products << ',' << e.value.significand << ',' << e.value.exponent << ','
            << text(decimal(e.value)) << ',' << e.rounded_sum.significand << ','
            << e.rounded_sum.exponent << ',' << e.rounded_products_finite << ','
            << text(decimal(e.absolute_sum)) << ',';
        if (e.min_nonzero_product.significand == 0) out << "NA";
        else out << text(decimal(e.min_nonzero_product));
        out << ',' << text(decimal(e.max_product)) << ',';
        if (e.value.significand == 0) out << "NA";
        else out << text(Decimal(decimal(e.absolute_sum) / decimal(magnitude(e.value))));
        out << '\n';
    }
}

template <typename T>
ErrorSummary analyze_errors(
    const std::vector<ReferenceEntry>& reference, const std::vector<T>& values,
    std::ostream* details, const std::string& prefix)
{
    ErrorSummary summary;
    summary.samples = reference.size();
    ExactBinary max_absolute, max_rel_num, max_rel_den{1, 0};
    bool have_relative = false;
    for (const auto& entry : reference) {
        if (entry.slot >= values.size()) throw std::invalid_argument("Reference slot out of range.");
        const T observed = values[entry.slot];
        if (entry.value.significand == 0) ++summary.zero_references;
        if (!std::isfinite(observed)) {
            ++summary.nonfinite;
            if (details) *details << prefix << ',' << entry.row << ',' << entry.column << ','
                << std::setprecision(17) << observed << ',' << value_bits(observed)
                << ",NA,NA,NA,NA,NA\n";
            continue;
        }
        const auto signed_error = difference(exact_from_double(static_cast<double>(observed)), entry.value);
        const auto absolute_error = magnitude(signed_error);
        if (absolute_error.significand != 0) ++summary.nonzero_errors;
        if (greater(absolute_error, max_absolute)) {
            max_absolute = absolute_error;
            summary.worst_row = std::to_string(entry.row);
            summary.worst_column = std::to_string(entry.column);
        }
        const auto denominator = magnitude(entry.value);
        if (denominator.significand != 0 && (!have_relative ||
            greater(product(absolute_error, max_rel_den), product(max_rel_num, denominator)))) {
            max_rel_num = absolute_error;
            max_rel_den = denominator;
            have_relative = true;
        }
        if (details) {
            *details << prefix << ',' << entry.row << ',' << entry.column << ','
                << std::setprecision(17) << observed << ',' << value_bits(observed) << ','
                << text(decimal(signed_error)) << ',' << text(decimal(absolute_error)) << ',';
            if (denominator.significand == 0) *details << "NA";
            else *details << text(Decimal(decimal(absolute_error) / decimal(denominator)));
            if (entry.rounded_products_finite) {
                *details << ',' << text(decimal(difference(entry.rounded_sum, entry.value)))
                    << ',' << text(decimal(difference(exact_from_double(static_cast<double>(observed)), entry.rounded_sum)));
            } else *details << ",NA,NA";
            *details << '\n';
        }
    }
    summary.max_absolute_error = text(decimal(max_absolute));
    if (have_relative) summary.max_relative_error = text(Decimal(decimal(max_rel_num) / decimal(max_rel_den)));
    if (summary.samples == 0 || summary.nonfinite == summary.samples) summary.max_absolute_error = "NA";
    return summary;
}

template std::vector<ReferenceEntry> build_reference<float>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<float>&, const SymbolicResult&, Offset);
template std::vector<ReferenceEntry> build_reference<double>(const CSRMatrix&, const CSRMatrix&, const NumericInputs<double>&, const SymbolicResult&, Offset);
template ErrorSummary analyze_errors<float>(const std::vector<ReferenceEntry>&, const std::vector<float>&, std::ostream*, const std::string&);
template ErrorSummary analyze_errors<double>(const std::vector<ReferenceEntry>&, const std::vector<double>&, std::ostream*, const std::string&);
}
