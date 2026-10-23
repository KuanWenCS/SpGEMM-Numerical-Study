#include "exact_binary.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace study {
namespace {

static_assert(
    sizeof(double) == sizeof(std::uint64_t) &&
    std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::radix == 2 &&
    std::numeric_limits<double>::digits == 53 &&
    std::numeric_limits<double>::max_exponent == 1024 &&
    std::numeric_limits<double>::min_exponent == -1021,
    "This decoder requires IEEE 754 binary64 double.");

// Shift the magnitude and then restore the sign.
ExactInteger shift_left_signed(
    const ExactInteger& value,
    int shift)
{
    if (shift < 0) {
        throw std::invalid_argument(
            "Exact alignment requires a nonnegative shift.");
    }

    const bool negative = value < 0;

    ExactInteger magnitude = value;
    if (negative) {
        magnitude = -magnitude;
    }

    magnitude <<= static_cast<unsigned int>(shift);

    if (negative) {
        magnitude = -magnitude;
    }

    return magnitude;
}

} // namespace

BinaryTerm decode_binary(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const bool negative = (bits >> 63) != 0;

    const std::uint64_t exponent_field =
        (bits >> 52) & 0x7ffULL;

    const std::uint64_t fraction =
        bits & ((std::uint64_t{1} << 52) - 1);

    if (exponent_field == 0x7ffULL) {
        throw std::invalid_argument(
            "Cannot decode NaN or infinity as a finite exact value.");
    }

    std::uint64_t magnitude = 0;
    int exponent = 0;

    if (exponent_field == 0) {
        // Subnormal values have no implicit leading bit.
        magnitude = fraction;
        exponent = -1074;
    } else {
        magnitude = (std::uint64_t{1} << 52) | fraction;
        exponent = static_cast<int>(exponent_field) - 1075;
    }

    if (magnitude == 0) {
        return {0, 0};
    }

    std::int64_t significand =
        static_cast<std::int64_t>(magnitude);

    if (negative) {
        significand = -significand;
    }

    return {significand, exponent};
}

ExactBinary exact_from_double(double value)
{
    const BinaryTerm term = decode_binary(value);

    return {
        ExactInteger(term.significand),
        term.exponent
    };
}

ExactBinary multiply_exact(
    const BinaryTerm& left,
    const BinaryTerm& right)
{
    if (left.significand == 0 || right.significand == 0) {
        return {};
    }

    ExactBinary result;
    result.significand = left.significand;
    result.significand *= right.significand;
    result.exponent = left.exponent + right.exponent;

    return result;
}

void accumulate_exact(
    ExactBinary& accumulator,
    const ExactBinary& term)
{
    if (term.significand == 0) {
        return;
    }

    if (accumulator.significand == 0) {
        accumulator = term;
        return;
    }

    const int common_exponent =
        accumulator.exponent < term.exponent
            ? accumulator.exponent
            : term.exponent;

    ExactInteger aligned_accumulator = shift_left_signed(
        accumulator.significand,
        accumulator.exponent - common_exponent);

    const ExactInteger aligned_term = shift_left_signed(
        term.significand,
        term.exponent - common_exponent);

    accumulator.significand =
        aligned_accumulator + aligned_term;
    accumulator.exponent = common_exponent;

    if (accumulator.significand == 0) {
        accumulator.exponent = 0;
    }
}

bool equal_exact(
    const ExactBinary& left,
    const ExactBinary& right)
{
    ExactBinary difference = left;
    ExactBinary negative_right = right;
    negative_right.significand = -negative_right.significand;

    accumulate_exact(difference, negative_right);

    return difference.significand == 0;
}

} // namespace study