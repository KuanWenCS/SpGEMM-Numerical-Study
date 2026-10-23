# SpGEMM Numerical Study

Compare CPU/GPU FP32/FP64 SpGEMM under fixed CSR order and parallel atomic accumulation. Exact references, per-entry product characteristics, repeated-run variation, diagnostic order traces, and stage timings share one experiment entry point.

## Build

Requires C++17, Python 3.12+, and Boost 1.88.0 headers. CUDA builds additionally require CUDA 12.0 and a compatible NVIDIA GPU. The default CUDA target is `sm_75`.

```bash
python3 scripts/setup_boost.py
make -j2 cpu
make test-cpu
```

On the server:

```bash
make -j2 cuda
CUDA_VISIBLE_DEVICES=1 make test-gpu
```

Choose an available GPU yourself; `1` is only an example. `make all` builds CPU programs and also CUDA programs when `nvcc` is found. `make test` similarly includes GPU tests when CUDA is available. GPU selection for tests is `make test-gpu GPU=1`.

`make test-gpu` runs both controlled GPU tests (including subnormal probes) and the independent Fraction integration oracle against CUDA results.

The default host compiler is Clang on macOS and `/usr/bin/g++-12` on Linux. Override with `make CXX=...`. Host arithmetic uses `-fno-fast-math -ffp-contract=off`; CUDA adds `--fmad=false --ftz=false`. Force rebuilding with `make -B` after changing compiler options.

## First integrated run

CPU-only, using the included tiny controlled case:

```bash
python3 scripts/run_experiment.py \
  --binary build/study \
  --a src/tests/data/order_A.mtx \
  --p src/tests/data/order_P.mtx \
  --out results/runs/cpu_smoke \
  --backend cpu --repeat 3 --samples all --threads 4 --trace 0,0
```

CPU/GPU server smoke test:

```bash
python3 scripts/run_experiment.py \
  --binary build/study_cuda --gpu 1 \
  --a src/tests/data/order_A.mtx \
  --p src/tests/data/order_P.mtx \
  --out results/runs/cuda_smoke \
  --backend both --repeat 3 --samples all --threads 4 --trace 0,0
```

Generated `results/runs/` directories are ignored by Git; explicitly archive selected reports when preparing a research checkpoint.

Each output directory must be new. Existing results are never overwritten. Progress is written to `stderr.log`; `manifest.json` records the command, input SHA-256 hashes, Git commit and dirty state, platform, GPU snapshot, process wall time, and final status. A dirty run is explicitly recorded, not rejected.

## Dataset runs

Validate the smaller pair first:

```bash
python3 scripts/run_experiment.py \
  --binary build/study_cuda --gpu 1 \
  --a matrices/mesh_2d_h005.mtx \
  --p matrices/mesh_2d_h005_P.mtx \
  --out results/runs/h005 \
  --backend both --repeat 3 --samples 10000 --threads 8
```

For a quick timing run on the largest pair, start without exact-reference reports:

```bash
python3 scripts/run_experiment.py \
  --binary build/study_cuda --gpu 1 \
  --a matrices/mesh_2d_h001.mtx \
  --p matrices/mesh_2d_h001_P.mtx \
  --out results/runs/h001_timing \
  --backend both --repeat 1 --samples 0 --threads 8
```

Then use a reproducible reference sample:

```bash
python3 scripts/run_experiment.py \
  --binary build/study_cuda --gpu 1 \
  --a matrices/mesh_2d_h001.mtx \
  --p matrices/mesh_2d_h001_P.mtx \
  --out results/runs/h001 \
  --backend both --repeat 3 --samples 10000 --threads 8
```

`--samples all` computes references for every structural output entry. Full references and detailed reports for the largest pair can require many GB; they are opt-in. With `--samples N`, error summaries apply only to those deterministic, evenly spaced output slots. Baseline comparisons and repeated-run change/range summaries still inspect the complete output. `--samples 0` disables references and reports errors as `NA`.

Use `--precision fp32|fp64|both`, `--strategy serial|atomic|both`, and `--gpu-threads N` to narrow or change a run. Defaults: both precisions, both strategies, CPU backend, three repeats, 10,000 reference entries, eight CPU threads, 128 GPU threads per block. The configured CPU thread count applies to the atomic implementation; the serial CPU baseline uses one thread.

## Reports

Files inside `artifacts/`:

| File | Meaning |
|---|---|
| `metadata.txt`, `matrices.csv`, `conversion.csv` | Build/runtime settings, input structure/value statistics, precision-conversion changes |
| `reference_fp32.csv`, `reference_fp64.csv` | Exact integer/exponent references and per-entry product/row/column characteristics |
| `summary.csv` | Per-run differences from CPU serial and errors against the precision-specific exact reference |
| `errors.csv` | Selected-entry signed/absolute/relative errors and multiplication/accumulation error decomposition |
| `timings.csv` | Loading, structure, conversion, reference, execution, and error-analysis timings |
| `variability.csv`, `repeat_samples.csv` | Complete-output change/range summary and sampled-entry ranges/distinct bit patterns |
| `trace_*.csv`, `trace_summary.csv` | Optional instrumented atomic order, products, before/after values, and replay verification |
| `status.txt` | `complete` or a failure message |

Non-finite values are counted separately. Relative errors and cancellation condition are `NA` when the reference is zero. Decimal error fields are approximations printed to 25 significant digits; reference significand/exponent fields preserve the exact value.

## Code map

- `src/core/matrix_io.cpp`: Matrix Market loading and CSR validation.
- `src/core/spgemm_symbolic.cpp`: output structure and path counts.
- `include/spgemm_numeric.hpp`: typed inputs and serial CPU calculation.
- `src/core/experiment_cpu.cpp`: CPU parallel atomics and diagnostic trace.
- `src/core/spgemm_gpu.cu`: serial/atomic GPU calculation and GPU replay.
- `src/core/exact_binary.cpp`, `src/core/error_analysis.cpp`: exact reference arithmetic and error analysis.
- `src/apps/run_study.cpp`: experiment orchestration and reports.
- `src/tests/`: controlled correctness tests; the Python integration test independently checks references with `fractions.Fraction`.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for interpretation and limitations, and [docs/PROGRESS.md](docs/PROGRESS.md) for validation status. Original examples and inspection programs remain available.
