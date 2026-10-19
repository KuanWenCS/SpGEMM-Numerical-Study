#include <algorithm>
#include <cstdio>
#include <exception>

#include "matrix_io.hpp"

namespace {

void print_summary(
    const char* name, const study::MatrixInput& input)
{
    const auto& matrix = input.matrix;

    const auto explicit_zeros = std::count(
        matrix.values.begin(), matrix.values.end(), 0.0);

    std::printf(
        "%s,%llu,%llu,%llu,%llu,%llu,%s\n",
        name,
        static_cast<unsigned long long>(matrix.rows),
        static_cast<unsigned long long>(matrix.cols),
        static_cast<unsigned long long>(
            input.source.stored_entries),
        static_cast<unsigned long long>(matrix.values.size()),
        static_cast<unsigned long long>(explicit_zeros),
        input.source.symmetric ? "symmetric" : "general");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(
            stderr, "Usage: %s A.mtx P.mtx\n", argv[0]);
        return 1;
    }

    try {
        std::fprintf(stderr, "Reading A...\n");
        const auto A = study::load_matrix_market(argv[1]);

        std::fprintf(stderr, "Reading P...\n");
        const auto P = study::load_matrix_market(argv[2]);

        if (A.matrix.cols != P.matrix.rows) {
            std::fprintf(
                stderr, "FAIL: A.cols does not equal P.rows.\n");
            return 1;
        }

        std::printf(
            "matrix,rows,cols,file_entries,"
            "csr_entries,explicit_zeros,source_symmetry\n");

        print_summary("A", A);
        print_summary("P", P);

        std::fprintf(
            stderr,
            "PASS: inputs are valid and compatible for A * P.\n"
            "CSR order: row ascending, column ascending.\n"
            "Duplicate policy: reject. Explicit zeros: retain.\n");

        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}