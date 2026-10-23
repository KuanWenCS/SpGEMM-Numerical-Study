#ifndef EXACT_BINARY_HPP
#define EXACT_BINARY_HPP

#include <cstdint>

#include <boost/multiprecision/cpp_int.hpp>

namespace study {

using ExactInteger = boost::multiprecision::cpp_int;

// Exact representation of one finite binary64 input.
struct BinaryTerm {
    std::int64_t significand = 0;
    int exponent = 0;
};

// Exact value = significand * 2^exponent.
struct ExactBinary {
    ExactInteger significand = 0;
    int exponent = 0;
};

// Reject NaN and infinity. Both signed zeros become exact zero.
BinaryTerm decode_binary(double value);

ExactBinary exact_from_double(double value);

ExactBinary multiply_exact(
    const BinaryTerm& left,
    const BinaryTerm& right);

// Update an accumulator without floating-point rounding.
void accumulate_exact(
    ExactBinary& accumulator,
    const ExactBinary& term);

// Compare mathematical values, allowing different representations.
bool equal_exact(
    const ExactBinary& left,
    const ExactBinary& right);

} // namespace study

#endif