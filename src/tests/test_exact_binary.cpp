#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>

#include "exact_binary.hpp"

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_decode()
{
    require(
        study::equal_exact(
            study::exact_from_double(1.0),
            study::ExactBinary{1, 0}),
        "Incorrect representation of one.");

    require(
        study::equal_exact(
            study::exact_from_double(-0.75),
            study::ExactBinary{-3, -2}),
        "Incorrect negative fractional representation.");

    require(
        study::equal_exact(
            study::exact_from_double(-0.0),
            study::ExactBinary{}),
        "Signed zero must represent exact zero.");

    require(
        study::equal_exact(
            study::exact_from_double(
                std::numeric_limits<double>::denorm_min()),
            study::ExactBinary{1, -1074}),
        "Incorrect smallest subnormal.");

    require(
        study::equal_exact(
            study::exact_from_double(
                std::numeric_limits<double>::min()),
            study::ExactBinary{1, -1022}),
        "Incorrect smallest normal.");

    std::puts("decode: PASS");
}

void test_order_independence()
{
    const double values[] = {33554432.0, -33554432.0, 1.0};

    study::ExactBinary forward;
    study::ExactBinary reverse;

    for (int k = 0; k < 3; ++k) {
        study::accumulate_exact(
            forward, study::exact_from_double(values[k]));

        study::accumulate_exact(
            reverse, study::exact_from_double(values[2 - k]));
    }

    const study::ExactBinary expected{1, 0};

    require(
        study::equal_exact(forward, expected) &&
        study::equal_exact(reverse, expected),
        "Exact sums must be independent of order.");

    std::puts("order_independence: PASS");
}

void test_product_rounding()
{
    const double value = std::nextafter(1.0, 2.0);
    const double rounded_product = value * value;

    study::ExactBinary difference = study::multiply_exact(
        study::decode_binary(value),
        study::decode_binary(value));

    auto negative_rounded =
        study::exact_from_double(rounded_product);
    negative_rounded.significand =
        -negative_rounded.significand;

    study::accumulate_exact(difference, negative_rounded);

    // (1 + 2^-52)^2 contains a term lost by binary64 multiplication.
    require(
        study::equal_exact(
            difference, study::ExactBinary{1, -104}),
        "Exact multiplication did not preserve the lost product term.");

    std::puts("product_rounding: PASS");
}

void test_below_binary64_range()
{
    const auto result = study::multiply_exact(
        study::decode_binary(
            std::numeric_limits<double>::denorm_min()),
        study::decode_binary(0.5));

    require(
        study::equal_exact(
            result, study::ExactBinary{1, -1075}),
        "Exact reference lost a value below binary64 range.");

    std::puts("below_binary64_range: PASS");
}

void test_precision_specific_inputs()
{
    const float fp32_input = 0.1f;
    const double fp64_input = 0.1;

    const auto fp32_exact = study::exact_from_double(
        static_cast<double>(fp32_input));

    const auto fp64_exact =
        study::exact_from_double(fp64_input);

    require(
        !study::equal_exact(fp32_exact, fp64_exact),
        "Distinct typed inputs must remain distinct.");

    std::puts("precision_specific_inputs: PASS");
}

void test_nonfinite_rejection()
{
    const double values[] = {
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()
    };

    for (const double value : values) {
        bool rejected = false;

        try {
            study::decode_binary(value);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }

        require(rejected, "Non-finite input must be rejected.");
    }

    std::puts("nonfinite_rejection: PASS");
}

} // namespace

int main()
{
    try {
        test_decode();
        test_order_independence();
        test_product_rounding();
        test_below_binary64_range();
        test_precision_specific_inputs();
        test_nonfinite_rejection();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}