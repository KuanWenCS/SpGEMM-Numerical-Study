#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#include "spgemm_symbolic.hpp"

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_hand_case()
{
    const study::CSRMatrix A{
        2, 3,
        {0, 2, 4},
        {0, 2, 1, 2},
        {1, 2, 3, 4}
    };

    const study::CSRMatrix P{
        3, 2,
        {0, 1, 2, 4},
        {0, 1, 0, 1},
        {5, 7, 6, 8}
    };

    const auto result = study::build_output_pattern(A, P);

    require(
        result.pattern.row_ptr ==
            std::vector<study::Offset>{0, 2, 4},
        "Incorrect output row offsets.");

    require(
        result.pattern.col_idx ==
            std::vector<study::Index>{0, 1, 0, 1},
        "Incorrect output columns.");

    require(
        result.products_per_entry ==
            std::vector<study::Offset>{2, 1, 1, 2},
        "Incorrect per-entry product counts.");

    require(result.total_products == 6, "Incorrect total.");

    std::puts("hand_case: PASS");
}

void test_zero_and_cancellation()
{
    study::CSRMatrix A{
        1, 2,
        {0, 2},
        {0, 1},
        {0, 1}
    };

    const study::CSRMatrix P{
        2, 1,
        {0, 1, 2},
        {0, 0},
        {1, 1}
    };

    const auto zero_result =
        study::build_output_pattern(A, P);

    require(
        zero_result.products_per_entry ==
            std::vector<study::Offset>{2},
        "Explicit-zero paths must be retained.");

    A.values = {1, -1};

    const auto cancellation_result =
        study::build_output_pattern(A, P);

    require(
        cancellation_result.pattern.col_idx ==
            std::vector<study::Index>{0},
        "Potential cancellation must not remove structure.");

    require(
        cancellation_result.products_per_entry ==
            std::vector<study::Offset>{2},
        "Incorrect cancellation-case path count.");

    std::puts("zero_and_cancellation_structure: PASS");
}

void test_empty_output_rows()
{
    const study::CSRMatrix A{
        3, 2,
        {0, 1, 1, 2},
        {0, 1},
        {1, 1}
    };

    const study::CSRMatrix P{
        2, 2,
        {0, 0, 1},
        {1},
        {1}
    };

    const auto result = study::build_output_pattern(A, P);

    require(
        result.pattern.row_ptr ==
            std::vector<study::Offset>{0, 0, 0, 1},
        "Empty output rows were not preserved.");

    require(
        result.pattern.col_idx ==
            std::vector<study::Index>{1},
        "Incorrect column after empty rows.");

    require(result.total_products == 1, "Incorrect total.");

    std::puts("empty_output_rows: PASS");
}

void test_no_output_rows()
{
    const study::CSRMatrix A{
        0, 1, {0}, {}, {}
    };

    const study::CSRMatrix P{
        1, 1, {0, 1}, {0}, {1}
    };

    const auto result = study::build_output_pattern(A, P);

    require(
        result.pattern.row_ptr ==
            std::vector<study::Offset>{0},
        "Incorrect structure for zero output rows.");

    require(result.pattern.col_idx.empty(), "Expected no entries.");
    require(result.total_products == 0, "Expected zero products.");

    std::puts("no_output_rows: PASS");
}

} // namespace

int main()
{
    try {
        test_hand_case();
        test_zero_and_cancellation();
        test_empty_output_rows();
        test_no_output_rows();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}