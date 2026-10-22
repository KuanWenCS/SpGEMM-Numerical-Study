#include <exception>
#include <iomanip>
#include <iostream>

#include "matrix_analysis.hpp"
#include "matrix_io.hpp"

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

        std::cerr << "Counting structural products...\n";
        const auto result = study::analyze_workload(
            A.matrix, P.matrix);

        std::cout
            << "output_rows,output_cols,total_products,"
            << "mean_products_per_row,max_products_per_row,"
            << "max_row_index,zero_product_rows\n";

        std::cout
            << A.matrix.rows << ','
            << P.matrix.cols << ','
            << result.total_products << ','
            << std::setprecision(17)
            << result.mean_products_per_row << ','
            << result.max_products_per_row << ',';

        if (result.max_row_index.has_value()) {
            std::cout << *result.max_row_index;
        } else {
            std::cout << "NA";
        }

        std::cout
            << ',' << result.zero_product_rows << '\n';

        std::cerr
            << "PASS: workload analysis completed.\n"
            << "Explicit-zero paths are included.\n"
            << "Row indices are zero-based.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}