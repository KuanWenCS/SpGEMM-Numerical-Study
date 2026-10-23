#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "error_analysis.hpp"
#include "experiment.hpp"

using namespace study;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void tests()
{
    const CSRMatrix A{2, 3, {0, 2, 4}, {0, 2, 1, 2}, {1, 2, 3, 4}};
    const CSRMatrix P{3, 2, {0, 1, 2, 4}, {0, 1, 0, 1}, {5, 7, 6, 8}};
    const auto symbolic = build_output_pattern(A, P);
    const auto inputs = prepare_numeric_inputs<T>(A, P);
    ExecutionOptions options;
    options.threads = 4;
    for (auto strategy : {Strategy::serial, Strategy::atomic}) {
        const auto result = execute_cpu(A, P, inputs, symbolic.pattern, strategy, options);
        require(result.values == std::vector<T>({17, 16, 24, 53}), "Integer-valued multiplication failed.");
    }
    const auto reference = build_reference(A, P, inputs, symbolic, 100);
    require(reference.size() == 4 && reference[0].products == 2, "Reference path count failed.");
    const auto correct = multiply_cpu_serial(A, P, inputs, symbolic.pattern);
    require(analyze_errors(reference, correct, nullptr, "").nonzero_errors == 0, "Exact hand case has an error.");
    auto wrong = correct;
    wrong[0] += 1;
    const auto error = analyze_errors(reference, wrong, nullptr, "");
    require(error.nonzero_errors == 1 && error.worst_row == "0" && error.worst_column == "0", "Error location failed.");
    const auto sampled = build_reference(A, P, inputs, symbolic, 2);
    require(sampled.size() == 2 && sampled[0].slot == 0 && sampled[1].slot == 3, "Deterministic sampling failed.");
    require(build_reference(A, P, inputs, symbolic, 0).empty(), "No-reference mode failed.");

    const CSRMatrix cancel_A{1, 3, {0, 3}, {0, 1, 2}, {33554432, 1, -33554432}};
    const CSRMatrix cancel_P{3, 1, {0, 1, 2, 3}, {0, 0, 0}, {1, 1, 1}};
    const auto cs = build_output_pattern(cancel_A, cancel_P);
    const auto ci = prepare_numeric_inputs<T>(cancel_A, cancel_P);
    const auto cr = build_reference(cancel_A, cancel_P, ci, cs, 1);
    require(equal_exact(cr[0].value, ExactBinary{1, 0}), "Cancellation reference is not one.");
    const auto cv = multiply_cpu_serial(cancel_A, cancel_P, ci, cs.pattern);
    require(analyze_errors(cr, cv, nullptr, "").nonzero_errors == (std::is_same_v<T, float> ? 1 : 0), "Precision-specific error failed.");
    options.trace_slot = 0;
    options.trace_capacity = 3;
    const auto trace = execute_cpu(cancel_A, cancel_P, ci, cs.pattern, Strategy::atomic, options);
    require(trace.trace.size() == 3 && value_bits(replay_cpu(trace.trace)) == value_bits(trace.values[0]), "Trace replay failed.");
    auto corrupt = trace.trace;
    corrupt[0].before = T{2};
    bool rejected = false;
    try { replay_cpu(corrupt); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Corrupt trace accepted.");

    const CSRMatrix zero_A{1, 2, {0, 2}, {0, 1}, {1, -1}};
    const CSRMatrix zero_P{2, 1, {0, 1, 2}, {0, 0}, {1, 1}};
    const auto zs = build_output_pattern(zero_A, zero_P);
    const auto zi = prepare_numeric_inputs<T>(zero_A, zero_P);
    const auto zr = build_reference(zero_A, zero_P, zi, zs, 1);
    const auto ze = analyze_errors(zr, std::vector<T>{1}, nullptr, "");
    require(ze.zero_references == 1 && ze.max_relative_error == "NA", "Zero reference policy failed.");
    const auto ne = analyze_errors(zr, std::vector<T>{std::numeric_limits<T>::infinity()}, nullptr, "");
    require(ne.nonfinite == 1, "Nonfinite accounting failed.");

    CSRMatrix many_A;
    many_A.rows = 1; many_A.cols = 256; many_A.row_ptr = {0, 256};
    CSRMatrix many_P;
    many_P.rows = 256; many_P.cols = 1;
    for (Index i = 0; i < 256; ++i) {
        many_A.col_idx.push_back(i); many_A.values.push_back(1);
        many_P.row_ptr.push_back(i); many_P.col_idx.push_back(0); many_P.values.push_back(1);
    }
    many_P.row_ptr.push_back(256);
    const auto ms = build_output_pattern(many_A, many_P);
    const auto mi = prepare_numeric_inputs<T>(many_A, many_P);
    options.trace_slot.reset();
    const auto many = execute_cpu(many_A, many_P, mi, ms.pattern, Strategy::atomic, options);
    require(many.values == std::vector<T>{256}, "Atomic updates were lost.");

    const CSRMatrix empty_A{2, 1, {0, 0, 0}, {}, {}};
    const CSRMatrix empty_P{1, 0, {0, 0}, {}, {}};
    const auto es = build_output_pattern(empty_A, empty_P);
    const auto ei = prepare_numeric_inputs<T>(empty_A, empty_P);
    require(execute_cpu(empty_A, empty_P, ei, es.pattern, Strategy::atomic, options).values.empty(), "Empty result failed.");
    require(build_reference(empty_A, empty_P, ei, es, 100).empty(), "Empty reference failed.");
    const CSRMatrix tiny_A{1, 1, {0, 1}, {0}, {static_cast<double>(std::numeric_limits<T>::denorm_min())}};
    const CSRMatrix tiny_P{1, 1, {0, 1}, {0}, {0.5}};
    const auto ts = build_output_pattern(tiny_A, tiny_P);
    const auto ti = prepare_numeric_inputs<T>(tiny_A, tiny_P);
    const auto tr = build_reference(tiny_A, tiny_P, ti, ts, 1);
    const auto tv = multiply_cpu_serial(tiny_A, tiny_P, ti, ts.pattern);
    require(tr[0].value.significand != 0 && tv[0] == T{0}, "Exact reference lost an underflowed product.");
    require(analyze_errors(tr, tv, nullptr, "").nonzero_errors == 1, "Underflow error not detected.");
    require(analyze_errors(std::vector<ReferenceEntry>{}, tv, nullptr, "").max_absolute_error == "NA", "Disabled reference reported zero error.");
    const CSRMatrix huge_A{1, 2, {0, 2}, {0, 1}, {static_cast<double>(std::numeric_limits<T>::max()), -static_cast<double>(std::numeric_limits<T>::max())}};
    const CSRMatrix huge_P{2, 1, {0, 1, 2}, {0, 0}, {2, 2}};
    const auto hs = build_output_pattern(huge_A, huge_P);
    const auto hi = prepare_numeric_inputs<T>(huge_A, huge_P);
    const auto hr = build_reference(huge_A, huge_P, hi, hs, 1);
    const auto hv = multiply_cpu_serial(huge_A, huge_P, hi, hs.pattern);
    require(hr[0].value.significand == 0 && !hr[0].rounded_products_finite, "Overflowing product reference failed.");
    require(analyze_errors(hr, hv, nullptr, "").nonfinite == 1, "Nonfinite arithmetic result not reported.");
    std::cout << (std::is_same_v<T, float> ? "FP32" : "FP64") << " experiment: PASS\n";
}

void reader_tests()
{
    for (const std::string text : {
        "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n1 1 2\n",
        "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1\n",
        "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n",
        "%%MatrixMarket matrix coordinate real general\n2 2 0\n1 1 1\n"}) {
        std::istringstream stream(text);
        bool rejected = false;
        try { read_matrix_market(stream); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Malformed input accepted.");
    }
    std::istringstream symmetric("%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 0\n2 1 -1\n");
    const auto input = read_matrix_market(symmetric);
    require(input.matrix.values.size() == 3 && input.matrix.values[0] == 0, "Symmetric expansion or explicit zero policy failed.");
    require(larger_range(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                         0, std::numeric_limits<double>::max()), "Large range comparison failed.");
    require(range_text(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()).find("inf") == std::string::npos,
            "Finite exact range overflowed its decimal report.");
    std::cout << "reader and range regression: PASS\n";
}
}

int main()
{
    try { tests<float>(); tests<double>(); reader_tests(); return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
