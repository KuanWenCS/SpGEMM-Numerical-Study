#include "matrix_io.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace study {
namespace {

// Temporary coordinate-format entry used during CSR construction.
struct Entry {
    Index row;
    Index col;
    double value;
};

Offset parse_unsigned(const std::string& text)
{
    Offset value = 0;

    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);

    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        throw std::runtime_error(
            "Invalid unsigned integer: " + text);
    }

    return value;
}

void require_end(std::istringstream& line)
{
    std::string extra;

    if (line >> extra) {
        throw std::runtime_error(
            "Unexpected extra token: " + extra);
    }
}

// Skip comments and blank lines.
bool next_data_line(std::istream& input, std::string& line)
{
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");

        if (first != std::string::npos && line[first] != '%') {
            return true;
        }
    }

    if (input.bad()) {
        throw std::runtime_error("Input stream read failure.");
    }

    return false;
}

// Sort coordinates, reject duplicates, and build CSR arrays.
CSRMatrix build_csr(
    Index rows, Index cols, std::vector<Entry> entries)
{
    std::sort(
        entries.begin(), entries.end(),
        [](const Entry& left, const Entry& right) {
            if (left.row != right.row) {
                return left.row < right.row;
            }

            return left.col < right.col;
        });

    for (std::size_t p = 1; p < entries.size(); ++p) {
        if (entries[p - 1].row == entries[p].row &&
            entries[p - 1].col == entries[p].col) {
            throw std::runtime_error(
                "Duplicate coordinate at row " +
                std::to_string(
                    static_cast<Offset>(entries[p].row) + 1) +
                ", column " +
                std::to_string(
                    static_cast<Offset>(entries[p].col) + 1));
        }
    }

    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.row_ptr.assign(
        static_cast<std::size_t>(rows) + 1, Offset{0});

    matrix.col_idx.reserve(entries.size());
    matrix.values.reserve(entries.size());

    for (const auto& entry : entries) {
        ++matrix.row_ptr[
            static_cast<std::size_t>(entry.row) + 1];

        matrix.col_idx.push_back(entry.col);
        matrix.values.push_back(entry.value);
    }

    for (std::size_t i = 1; i < matrix.row_ptr.size(); ++i) {
        matrix.row_ptr[i] += matrix.row_ptr[i - 1];
    }

    return matrix;
}

} // namespace

void validate_csr(const CSRMatrix& matrix)
{
    const std::size_t expected_offsets =
        static_cast<std::size_t>(matrix.rows) + 1;

    if (matrix.row_ptr.size() != expected_offsets ||
        matrix.row_ptr.front() != 0) {
        throw std::runtime_error("Invalid CSR row offsets.");
    }

    if (matrix.col_idx.size() != matrix.values.size() ||
        matrix.row_ptr.back() != matrix.values.size()) {
        throw std::runtime_error("Inconsistent CSR array lengths.");
    }

    for (std::size_t i = 0; i < matrix.rows; ++i) {
        const Offset begin = matrix.row_ptr[i];
        const Offset end = matrix.row_ptr[i + 1];

        if (begin > end || end > matrix.values.size()) {
            throw std::runtime_error(
                "Invalid CSR row interval.");
        }

        for (Offset p = begin; p < end; ++p) {
            if (matrix.col_idx[p] >= matrix.cols) {
                throw std::runtime_error(
                    "CSR column index out of range.");
            }

            if (!std::isfinite(matrix.values[p])) {
                throw std::runtime_error(
                    "Non-finite CSR value.");
            }

            if (p > begin &&
                matrix.col_idx[p - 1] >= matrix.col_idx[p]) {
                throw std::runtime_error(
                    "CSR columns must be strictly increasing.");
            }
        }
    }
}

MatrixInput read_matrix_market(std::istream& input)
{
    std::string line;

    if (!std::getline(input, line)) {
        throw std::runtime_error("Missing Matrix Market header.");
    }

    std::istringstream header(line);
    std::string banner, object, format, field, symmetry;

    if (!(header >> banner >> object >> format
                 >> field >> symmetry)) {
        throw std::runtime_error("Invalid Matrix Market header.");
    }

    require_end(header);

    if (banner != "%%MatrixMarket" ||
        object != "matrix" ||
        format != "coordinate" ||
        field != "real" ||
        (symmetry != "general" && symmetry != "symmetric")) {
        throw std::runtime_error(
            "Supported formats: coordinate real general/symmetric.");
    }

    if (!next_data_line(input, line)) {
        throw std::runtime_error("Missing matrix dimensions.");
    }

    std::istringstream dimensions(line);
    std::string rows_text, cols_text, count_text;

    if (!(dimensions >> rows_text >> cols_text >> count_text)) {
        throw std::runtime_error("Invalid matrix dimensions.");
    }

    require_end(dimensions);

    const Offset rows = parse_unsigned(rows_text);
    const Offset cols = parse_unsigned(cols_text);
    const Offset stored_entries = parse_unsigned(count_text);
    const bool symmetric = symmetry == "symmetric";

    if (rows > std::numeric_limits<Index>::max() ||
        cols > std::numeric_limits<Index>::max()) {
        throw std::runtime_error(
            "Matrix dimensions exceed the index type.");
    }

    if (symmetric && rows != cols) {
        throw std::runtime_error(
            "A symmetric matrix must be square.");
    }

    std::vector<Entry> entries;
    const Offset capacity_factor = symmetric ? 2 : 1;

    if (stored_entries >
        entries.max_size() / capacity_factor) {
        throw std::runtime_error(
            "Declared entry count exceeds vector capacity.");
    }

    entries.reserve(
        static_cast<std::size_t>(
            stored_entries * capacity_factor));

    for (Offset p = 0; p < stored_entries; ++p) {
        if (!next_data_line(input, line)) {
            throw std::runtime_error(
                "Fewer entries than declared in the header.");
        }

        std::istringstream record(line);
        std::string row_text, col_text;
        double value = 0;

        if (!(record >> row_text >> col_text >> value)) {
            throw std::runtime_error(
                "Invalid entry at record " +
                std::to_string(p + 1));
        }

        require_end(record);

        const Offset row = parse_unsigned(row_text);
        const Offset col = parse_unsigned(col_text);

        if (row == 0 || row > rows ||
            col == 0 || col > cols) {
            throw std::runtime_error(
                "Index out of range at record " +
                std::to_string(p + 1));
        }

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Non-finite value at record " +
                std::to_string(p + 1));
        }

        const Index i = static_cast<Index>(row - 1);
        const Index j = static_cast<Index>(col - 1);

        entries.push_back({i, j, value});

        // Expand off-diagonal symmetric entries exactly once.
        if (symmetric && i != j) {
            entries.push_back({j, i, value});
        }
    }

    if (next_data_line(input, line)) {
        throw std::runtime_error(
            "More entries than declared in the header.");
    }

    MatrixInput result;
    result.source = {stored_entries, symmetric};

    result.matrix = build_csr(
        static_cast<Index>(rows),
        static_cast<Index>(cols),
        std::move(entries));

    validate_csr(result.matrix);
    return result;
}

MatrixInput load_matrix_market(const std::string& path)
{
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    try {
        return read_matrix_market(input);
    } catch (const std::exception& error) {
        throw std::runtime_error(path + ": " + error.what());
    }
}

} // namespace study