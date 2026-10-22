#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "matrix_analysis.hpp"
#include "matrix_io.hpp"
#include "spgemm_symbolic.hpp"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0] << " A.mtx P.mtx\n";
        return 1;
    }

    try {
        std::cerr << "Reading A...\n";
        const auto A = study::load_matrix_market(argv[1]);

        std::cerr << "Reading P...\n";
        const auto P = study::load_matrix_market(argv[2]);

        const auto workload = study::analyze_workload(
            A.matrix, P.matrix);

        std::cerr << "Building sparse output structure...\n";
        const auto result = study::build_output_pattern(
            A.matrix, P.matrix);

        if (result.total_products != workload.total_products) {
            throw std::runtime_error(
                "Symbolic and workload product totals differ.");
        }

        // Check path conservation separately for each output row.
        for (std::size_t i = 0; i < A.matrix.rows; ++i) {
            study::Offset row_products = 0;

            for (study::Offset p = result.pattern.row_ptr[i];
                 p < result.pattern.row_ptr[i + 1]; ++p) {
                row_products += result.products_per_entry[p];
            }

            if (row_products != workload.products_per_row[i]) {
                throw std::runtime_error(
                    "Per-row product counts differ.");
            }
        }

        study::Offset max_products = 0;
        study::Offset single_product_entries = 0;

        for (const auto count : result.products_per_entry) {
            max_products = std::max(max_products, count);

            if (count == 1) {
                ++single_product_entries;
            }
        }

        const auto entries = result.pattern.col_idx.size();

        const double mean_products = entries == 0
            ? 0.0
            : static_cast<double>(result.total_products) /
              static_cast<double>(entries);

        std::cout
            << "output_rows,output_cols,structural_entries,"
            << "total_products,mean_products_per_entry,"
            << "max_products_per_entry,single_product_entries\n";

        std::cout
            << result.pattern.rows << ','
            << result.pattern.cols << ','
            << entries << ','
            << result.total_products << ','
            << std::setprecision(17) << mean_products << ','
            << max_products << ','
            << single_product_entries << '\n';

        std::cerr
            << "PASS: output structure built.\n"
            << "PASS: total and per-row product counts agree.\n"
            << "No floating-point multiplication was performed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}