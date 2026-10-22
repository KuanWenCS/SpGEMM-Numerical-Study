#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#include "matrix_analysis.hpp"

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

    const auto result = study::analyze_workload(A, P);

    require(
        result.products_per_row ==
            std::vector<study::Offset>{3, 3},
        "Incorrect hand-case row counts.");

    require(result.total_products == 6, "Incorrect total.");
    require(result.mean_products_per_row == 3, "Incorrect mean.");
    require(result.max_products_per_row == 3, "Incorrect maximum.");
    require(result.max_row_index == 0, "Incorrect maximum row.");
    require(result.zero_product_rows == 0, "Incorrect zero count.");

    std::puts("hand_case: PASS");
}

void test_uneven_rows()
{
    const study::CSRMatrix A{
        3, 3,
        {0, 1, 2, 3},
        {0, 1, 2},
        {1, 1, 1}
    };

    const study::CSRMatrix P{
        3, 3,
        {0, 1, 3, 3},
        {0, 0, 1},
        {1, 1, 1}
    };

    const auto result = study::analyze_workload(A, P);

    require(
        result.products_per_row ==
            std::vector<study::Offset>{1, 2, 0},
        "Incorrect uneven row counts.");

    require(result.total_products == 3, "Incorrect total.");
    require(result.mean_products_per_row == 1, "Incorrect mean.");
    require(result.max_products_per_row == 2, "Incorrect maximum.");
    require(result.max_row_index == 1, "Incorrect maximum row.");
    require(result.zero_product_rows == 1, "Incorrect zero count.");

    std::puts("uneven_rows: PASS");
}

void test_explicit_zero()
{
    const study::CSRMatrix A{
        1, 1,
        {0, 1},
        {0},
        {0}
    };

    const study::CSRMatrix P{
        1, 2,
        {0, 2},
        {0, 1},
        {2, 3}
    };

    const auto result = study::analyze_workload(A, P);

    require(
        result.total_products == 2,
        "Explicit-zero paths must be counted.");

    std::puts("explicit_zero: PASS");
}

void test_no_output_rows()
{
    const study::CSRMatrix A{
        0, 1,
        {0},
        {},
        {}
    };

    const study::CSRMatrix P{
        1, 1,
        {0, 1},
        {0},
        {1}
    };

    const auto result = study::analyze_workload(A, P);

    require(result.products_per_row.empty(), "Expected no rows.");
    require(result.total_products == 0, "Expected zero products.");
    require(result.mean_products_per_row == 0, "Expected zero mean.");
    require(!result.max_row_index.has_value(), "Expected no maximum row.");

    std::puts("no_output_rows: PASS");
}

void test_incompatible_dimensions()
{
    const study::CSRMatrix A{
        1, 1, {0, 0}, {}, {}
    };

    const study::CSRMatrix P{
        2, 1, {0, 0, 0}, {}, {}
    };

    bool rejected = false;

    try {
        study::analyze_workload(A, P);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(rejected, "Incompatible dimensions must be rejected.");
    std::puts("incompatible_dimensions: PASS");
}

} // namespace

int main()
{
    try {
        test_hand_case();
        test_uneven_rows();
        test_explicit_zero();
        test_no_output_rows();
        test_incompatible_dimensions();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}