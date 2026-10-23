#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include "error_analysis.hpp"
#include "experiment.hpp"
#include "matrix_analysis.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace study;

namespace {
struct Options {
    std::string a, p, out;
    std::string backend = "cpu", precision = "both", strategy = "both";
    unsigned repeats = 3;
    Offset samples = 10000;
    ExecutionOptions execution;
    std::optional<std::pair<Index, Index>> trace;
};

Offset number(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected a nonnegative integer: " + text);
    }
    std::size_t end = 0;
    const auto value = std::stoull(text, &end);
    if (end != text.size()) throw std::invalid_argument("Invalid integer.");
    return value;
}

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--a") o.a = value;
        else if (key == "--p") o.p = value;
        else if (key == "--out") o.out = value;
        else if (key == "--backend") o.backend = value;
        else if (key == "--precision") o.precision = value;
        else if (key == "--strategy") o.strategy = value;
        else if (key == "--samples") o.samples = value == "all" ? std::numeric_limits<Offset>::max() : number(value);
        else if (key == "--repeat") {
            const auto n = number(value);
            if (!n || n > 1000) throw std::invalid_argument("Repeat must be in [1,1000].");
            o.repeats = static_cast<unsigned>(n);
        } else if (key == "--threads" || key == "--gpu-threads") {
            const auto n = number(value);
            if (!n || n > (key == "--threads" ? 256 : 1024)) throw std::invalid_argument("Invalid thread count.");
            if (key == "--threads") o.execution.threads = static_cast<unsigned>(n);
            else o.execution.gpu_threads = static_cast<unsigned>(n);
        } else if (key == "--trace") {
            const auto comma = value.find(',');
            if (comma == std::string::npos) throw std::invalid_argument("Trace format: row,column.");
            const auto row = number(value.substr(0, comma)), col = number(value.substr(comma + 1));
            if (row > std::numeric_limits<Index>::max() || col > std::numeric_limits<Index>::max()) {
                throw std::invalid_argument("Trace coordinate out of range.");
            }
            o.trace = {{static_cast<Index>(row), static_cast<Index>(col)}};
        } else throw std::invalid_argument("Unknown option: " + key);
    }
    if (o.a.empty() || o.p.empty() || o.out.empty()) throw std::invalid_argument("--a, --p and --out are required.");
    if (o.backend != "cpu" && o.backend != "gpu" && o.backend != "both") throw std::invalid_argument("Invalid backend.");
    if (o.precision != "fp32" && o.precision != "fp64" && o.precision != "both") throw std::invalid_argument("Invalid precision.");
    if (o.strategy != "serial" && o.strategy != "atomic" && o.strategy != "both") throw std::invalid_argument("Invalid strategy.");
    if (o.trace && o.strategy == "serial") throw std::invalid_argument("Trace requires atomic strategy.");
#ifndef STUDY_CUDA
    if (o.backend != "cpu") throw std::invalid_argument("This executable was built without CUDA. Use study_cuda on the server.");
#endif
    return o;
}

std::ofstream file(const fs::path& path)
{
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17);
    return stream;
}

double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void matrix_stats(std::ostream& out, const char* name, const MatrixInput& input)
{
    const auto& m = input.matrix;
    Offset zeros = 0, positive = 0, negative = 0, max_row = 0, max_column = 0;
    double min_abs = std::numeric_limits<double>::infinity(), max_abs = 0;
    std::vector<Offset> columns(m.cols, 0);
    for (std::size_t p = 0; p < m.values.size(); ++p) {
        const double v = m.values[p];
        if (v == 0) ++zeros;
        else {
            if (v > 0) ++positive; else ++negative;
            min_abs = std::min(min_abs, std::fabs(v));
            max_abs = std::max(max_abs, std::fabs(v));
        }
        max_column = std::max(max_column, ++columns[m.col_idx[p]]);
    }
    for (std::size_t i = 0; i < m.rows; ++i) max_row = std::max(max_row, m.row_ptr[i + 1] - m.row_ptr[i]);
    out << name << ',' << m.rows << ',' << m.cols << ',' << input.source.stored_entries << ','
        << m.values.size() << ',' << zeros << ',' << positive << ',' << negative << ',';
    if (std::isfinite(min_abs)) out << min_abs; else out << "NA";
    out << ',' << max_abs << ',' << max_row << ',' << max_column << '\n';
}

template <typename T>
ExecutionResult<T> execute(const std::string& backend, const CSRMatrix& A, const CSRMatrix& P,
    const NumericInputs<T>& inputs, const CSRPattern& pattern, Strategy strategy, const ExecutionOptions& options)
{
    const auto start = Clock::now();
    ExecutionResult<T> result;
    if (backend == "cpu") result = execute_cpu(A, P, inputs, pattern, strategy, options);
#ifdef STUDY_CUDA
    else result = execute_gpu(A, P, inputs, pattern, strategy, options);
#else
    else throw std::runtime_error("CUDA backend unavailable.");
#endif
    result.total_seconds = elapsed(start);
    return result;
}

template <typename T>
void run_precision(const Options& o, const CSRMatrix& A, const CSRMatrix& P,
    const SymbolicResult& symbolic, std::ostream& summary, std::ostream& errors,
    std::ostream& timings, std::ostream& variability, std::ostream& sample_ranges,
    std::ostream& conversion, std::ostream& trace_summary)
{
    const std::string precision = std::is_same_v<T, float> ? "fp32" : "fp64";
    const auto& pattern = symbolic.pattern;
    auto stage = [&](const std::string& phase, auto operation) {
        std::cerr << precision << ": " << phase << '\n';
        const auto start = Clock::now();
        auto result = operation();
        timings << precision << ",host,none,0," << phase << ',' << elapsed(start) << '\n' << std::flush;
        return result;
    };
    const auto inputs = stage("convert", [&] { return prepare_numeric_inputs<T>(A, P); });
    auto conversion_row = [&](const char* name, const std::vector<double>& original, const std::vector<T>& typed) {
        Offset changed = 0, became_zero = 0, subnormal = 0;
        for (std::size_t i = 0; i < typed.size(); ++i) {
            if (static_cast<double>(typed[i]) != original[i]) ++changed;
            if (original[i] != 0 && typed[i] == T{0}) ++became_zero;
            if (std::fpclassify(typed[i]) == FP_SUBNORMAL) ++subnormal;
        }
        conversion << precision << ',' << name << ',' << typed.size() << ',' << changed << ','
            << became_zero << ',' << subnormal << '\n';
    };
    conversion_row("A", A.values, inputs.a_values);
    conversion_row("P", P.values, inputs.p_values);
    const auto baseline = stage("cpu_serial_baseline", [&] {
        return multiply_cpu_serial(A, P, inputs, pattern);
    });
    const auto reference = stage("exact_reference", [&] {
        return build_reference(A, P, inputs, symbolic, o.samples);
    });
    auto ref_file = file(fs::path(o.out) / ("reference_" + precision + ".csv"));
    write_reference(ref_file, reference);
    std::vector<std::string> backends = o.backend == "both"
        ? std::vector<std::string>{"cpu", "gpu"} : std::vector<std::string>{o.backend};
    std::vector<Strategy> strategies;
    if (o.strategy != "atomic") strategies.push_back(Strategy::serial);
    if (o.strategy != "serial") strategies.push_back(Strategy::atomic);
    auto report = [&](const std::string& backend, const std::string& strategy, unsigned run,
                      const ExecutionResult<T>& result) {
        if (result.values.size() != baseline.size()) throw std::runtime_error("Output length mismatch.");
        Offset differences = 0, nonfinite = 0;
        for (std::size_t p = 0; p < baseline.size(); ++p) {
            if (value_bits(result.values[p]) != value_bits(baseline[p])) ++differences;
            if (!std::isfinite(result.values[p])) ++nonfinite;
        }
        const std::string prefix = precision + ',' + backend + ',' + strategy + ',' + std::to_string(run);
        const auto analysis_start = Clock::now();
        const auto error = analyze_errors(reference, result.values, &errors, prefix);
        timings << prefix << ",error_analysis," << elapsed(analysis_start) << '\n';
        summary << prefix << ',' << result.values.size() << ',' << differences << ',' << nonfinite << ','
            << error.samples << ',' << error.nonfinite << ',' << error.nonzero_errors << ','
            << error.zero_references << ',' << error.max_absolute_error << ',' << error.max_relative_error << ','
            << error.worst_row << ',' << error.worst_column << '\n' << std::flush;
        timings << prefix << ",prepare," << result.prepare_seconds << '\n'
            << prefix << ",compute," << result.compute_seconds << '\n'
            << prefix << ",download," << result.download_seconds << '\n'
            << prefix << ",call_wall," << result.total_seconds << '\n' << std::flush;
    };
    for (const auto& backend : backends) for (const auto strategy : strategies) {
        const std::string name = strategy == Strategy::serial ? "serial" : "atomic";
        std::vector<T> first, minimum, maximum;
        std::vector<unsigned char> changed, nonfinite_seen;
        std::vector<std::set<std::uint64_t>> sample_bits(reference.size());
        for (unsigned run = 1; run <= o.repeats; ++run) {
            std::cerr << precision << ' ' << backend << ' ' << name << " run " << run << '/' << o.repeats << '\n';
            const auto result = execute(backend, A, P, inputs, pattern, strategy, o.execution);
            report(backend, name, run, result);
            if (run == 1) {
                first = minimum = maximum = result.values;
                changed.assign(first.size(), 0);
                nonfinite_seen.assign(first.size(), 0);
            }
            for (std::size_t p = 0; p < first.size(); ++p) {
                if (value_bits(result.values[p]) != value_bits(first[p])) changed[p] = 1;
                if (!std::isfinite(result.values[p])) { nonfinite_seen[p] = 1; continue; }
                if (!std::isfinite(minimum[p])) minimum[p] = maximum[p] = result.values[p];
                minimum[p] = std::min(minimum[p], result.values[p]);
                maximum[p] = std::max(maximum[p], result.values[p]);
            }
            for (std::size_t r = 0; r < reference.size(); ++r) sample_bits[r].insert(value_bits(result.values[reference[r].slot]));
        }
        Offset changed_count = 0, nonfinite_count = 0;
        long double max_range = 0;
        bool have_finite_range = false;
        T range_minimum = 0, range_maximum = 0;
        for (std::size_t p = 0; p < first.size(); ++p) {
            changed_count += changed[p];
            nonfinite_count += nonfinite_seen[p];
            if (std::isfinite(minimum[p]) && std::isfinite(maximum[p])) {
                have_finite_range = true;
                const long double candidate = static_cast<long double>(maximum[p]) - static_cast<long double>(minimum[p]);
                if (candidate > max_range || (candidate == max_range && candidate != 0 &&
                    larger_range(minimum[p], maximum[p], range_minimum, range_maximum))) {
                    max_range = candidate;
                    range_minimum = minimum[p];
                    range_maximum = maximum[p];
                }
            }
        }
        variability << precision << ',' << backend << ',' << name << ',' << o.repeats << ','
            << first.size() << ',' << changed_count << ',' << nonfinite_count << ',' << (have_finite_range ? range_text(range_minimum, range_maximum) : "NA") << '\n' << std::flush;
        for (std::size_t r = 0; r < reference.size(); ++r) {
            const auto& e = reference[r];
            sample_ranges << precision << ',' << backend << ',' << name << ',' << e.row << ',' << e.column << ',';
            if (std::isfinite(minimum[e.slot])) sample_ranges << std::setprecision(17) << minimum[e.slot] << ',' << maximum[e.slot];
            else sample_ranges << "NA,NA";
            sample_ranges << ',' << sample_bits[r].size() << '\n';
        }
        if (strategy == Strategy::atomic && o.trace) {
            const Index row = o.trace->first, col = o.trace->second;
            if (row >= A.rows || col >= P.cols) throw std::invalid_argument("Trace coordinate out of bounds.");
            const auto begin = pattern.col_idx.begin() + pattern.row_ptr[row];
            const auto end = pattern.col_idx.begin() + pattern.row_ptr[row + 1];
            const auto found = std::lower_bound(begin, end, col);
            if (found == end || *found != col) throw std::invalid_argument("Trace coordinate is not structurally present.");
            const Offset slot = found - pattern.col_idx.begin();
            auto options = o.execution;
            options.trace_slot = slot;
            options.trace_capacity = symbolic.products_per_entry[slot];
            if (options.trace_capacity > 100000) throw std::invalid_argument("Trace exceeds 100000 updates.");
            std::cerr << "Diagnostic trace: " << precision << ' ' << backend << '\n';
            const auto diagnostic = execute(backend, A, P, inputs, pattern, strategy, options);
            if (diagnostic.trace.size() != options.trace_capacity) throw std::runtime_error("Trace length differs from path count.");
            std::set<std::pair<Offset, Offset>> origins;
            for (const auto& event : diagnostic.trace) {
                if (event.a_position < A.row_ptr[row] || event.a_position >= A.row_ptr[row + 1]) throw std::runtime_error("Invalid trace A origin.");
                const Index k = A.col_idx[event.a_position];
                if (event.p_position < P.row_ptr[k] || event.p_position >= P.row_ptr[k + 1] ||
                    P.col_idx[event.p_position] != col || !origins.insert({event.a_position, event.p_position}).second) {
                    throw std::runtime_error("Invalid or duplicate trace product origin.");
                }
            }
            T replayed = 0;
            if (backend == "cpu") replayed = replay_cpu(diagnostic.trace);
#ifdef STUDY_CUDA
            else replayed = replay_gpu(diagnostic.trace);
#endif
            if (value_bits(replayed) != value_bits(diagnostic.values[slot])) throw std::runtime_error("Trace final value mismatch.");
            auto trace_file = file(fs::path(o.out) / ("trace_" + precision + '_' + backend + ".csv"));
            trace_file << "sequence,a_position,p_position,worker,product,before,after,product_bits,before_bits,after_bits\n";
            for (std::size_t i = 0; i < diagnostic.trace.size(); ++i) {
                const auto& e = diagnostic.trace[i];
                trace_file << i << ',' << e.a_position << ',' << e.p_position << ',' << e.worker << ','
                    << e.product << ',' << e.before << ',' << e.after << ',' << value_bits(e.product) << ','
                    << value_bits(e.before) << ',' << value_bits(e.after) << '\n';
            }
            report(backend, "atomic_trace", 0, diagnostic);
            trace_summary << precision << ',' << backend << ',' << row << ',' << col << ','
                << diagnostic.trace.size() << ',' << value_bits(replayed) << ",1\n" << std::flush;
        }
    }
}
}

int main(int argc, char** argv)
{
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        std::cout << "--a A.mtx --p P.mtx --out NEW_DIRECTORY\n"
            "[--backend cpu|gpu|both] [--precision fp32|fp64|both]\n"
            "[--strategy serial|atomic|both] [--repeat 3] [--samples 10000|all|0]\n"
            "[--threads 8] [--gpu-threads 128] [--trace ROW,COLUMN]\n";
        return 0;
    }
    std::string output;
    try {
        const auto o = parse(argc, argv);
        if (fs::exists(o.out)) throw std::invalid_argument("Output directory already exists; choose a new path.");
        fs::create_directories(o.out);
        output = o.out;
        auto metadata = file(fs::path(o.out) / "metadata.txt");
        metadata << "compiler=" << __VERSION__ << '\n'
            << "a=" << fs::absolute(o.a).string() << '\n' << "p=" << fs::absolute(o.p).string() << '\n'
            << "backend=" << o.backend << "\nprecision=" << o.precision << "\nstrategy=" << o.strategy << '\n'
            << "cpu_threads=" << o.execution.threads << "\ngpu_threads=" << o.execution.gpu_threads << '\n'
            << "repeats=" << o.repeats << "\nsample_limit=" << o.samples << '\n'
            << "sampling=evenly_spaced_output_slots\nreference=exact_sum_of_exact_products_of_typed_inputs\n"
            << "input_policy=sorted_CSR,reject_duplicates,retain_explicit_zeros\n"
            << "rounding_mode=" << std::fegetround() << '\n'
            << "trace=additional_lock_instrumented_full_execution\n";
#ifdef STUDY_CUDA_FLAGS
        metadata << "cuda_build_flags=" << STUDY_CUDA_FLAGS << '\n';
#endif
#ifdef STUDY_BUILD_FLAGS
        metadata << "host_build_flags=" << STUDY_BUILD_FLAGS << '\n';
#endif
        if (std::fegetround() != FE_TONEAREST) throw std::runtime_error("Round-to-nearest host mode required.");
        if (const auto* visible = std::getenv("CUDA_VISIBLE_DEVICES")) metadata << "CUDA_VISIBLE_DEVICES=" << visible << '\n';
#ifdef STUDY_CUDA
        if (o.backend != "cpu") metadata << "gpu=" << gpu_description() << '\n';
#endif
        auto timings = file(fs::path(o.out) / "timings.csv");
        timings << "precision,backend,strategy,run,phase,seconds\n";
        auto start = Clock::now();
        std::cerr << "Reading A...\n";
        const auto A = load_matrix_market(o.a);
        timings << "NA,host,none,0,read_A," << elapsed(start) << '\n' << std::flush;
        start = Clock::now();
        std::cerr << "Reading P...\n";
        const auto P = load_matrix_market(o.p);
        timings << "NA,host,none,0,read_P," << elapsed(start) << '\n' << std::flush;
        auto stats = file(fs::path(o.out) / "matrices.csv");
        stats << "matrix,rows,columns,file_entries,csr_entries,zeros,positive,negative,min_nonzero_abs,max_abs,max_row_entries,max_column_entries\n";
        start = Clock::now();
        matrix_stats(stats, "A", A); matrix_stats(stats, "P", P);
        timings << "NA,host,none,0,input_statistics," << elapsed(start) << '\n';
        start = Clock::now();
        std::cerr << "Building output structure...\n";
        const auto symbolic = build_output_pattern(A.matrix, P.matrix);
        timings << "NA,host,none,0,symbolic," << elapsed(start) << '\n' << std::flush;
        metadata << "output_entries=" << symbolic.pattern.col_idx.size() << "\ntotal_products=" << symbolic.total_products << '\n';
        metadata.flush();
        auto summary = file(fs::path(o.out) / "summary.csv");
        summary << "precision,backend,strategy,run,output_entries,bitwise_diff_cpu_serial,nonfinite_outputs,sampled_entries,"
            "sampled_nonfinite,sampled_nonzero_errors,sampled_zero_references,max_abs_error,max_rel_error,worst_abs_row,worst_abs_column\n";
        auto errors = file(fs::path(o.out) / "errors.csv");
        errors << "precision,backend,strategy,run,row,column,value,value_bits,signed_error,abs_error,relative_error,product_error,accumulation_error\n";
        auto variability = file(fs::path(o.out) / "variability.csv");
        variability << "precision,backend,strategy,repeats,output_entries,entries_changed_from_first,entries_ever_nonfinite,max_finite_range\n";
        auto ranges = file(fs::path(o.out) / "repeat_samples.csv");
        ranges << "precision,backend,strategy,row,column,finite_min,finite_max,distinct_bit_patterns\n";
        auto conversion = file(fs::path(o.out) / "conversion.csv");
        conversion << "precision,matrix,entries,changed_from_parsed_fp64,became_zero,subnormal_entries\n";
        auto traces = file(fs::path(o.out) / "trace_summary.csv");
        traces << "precision,backend,row,column,updates,replayed_bits,replay_verified\n";
        if (o.precision != "fp64") run_precision<float>(o, A.matrix, P.matrix, symbolic, summary, errors, timings, variability, ranges, conversion, traces);
        if (o.precision != "fp32") run_precision<double>(o, A.matrix, P.matrix, symbolic, summary, errors, timings, variability, ranges, conversion, traces);
        auto status = file(fs::path(o.out) / "status.txt"); status << "complete\n";
        std::cerr << "Complete: " << o.out << '\n';
        return 0;
    } catch (const std::exception& e) {
        if (!output.empty()) { std::ofstream status(fs::path(output) / "status.txt"); status << "failed: " << e.what() << '\n'; }
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
