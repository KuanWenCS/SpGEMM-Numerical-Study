# SpGEMM Numerical Study — Progress

## Objective

Study numerical errors in CPU/GPU SpGEMM with FP32 and FP64, including accumulation strategies, actual accumulation order, and input/product characteristics. Repeated-run variation is supplementary evidence.

## Completed

- CUDA allocation, kernel execution, and result retrieval validated.
- Basic CPU and GPU SpGEMM prototypes validated on a hand-computed case in FP32 and FP64.
- Shared CSR input definition and hand-case fixture introduced.
- Controlled accumulation-order example validated:
  - FP32: forward = 1, reverse = 0.
  - FP64: forward = reverse = 1.
  - CPU and GPU agree in this example.
- Modular Matrix Market reader and input inspection driver implemented.
- Five reader checks passed: general, symmetric, duplicate rejection, out-of-range rejection, and missing-entry rejection.

## Dataset

Source: UIUC SpGEMM benchmark supplementary data

URL: https://zenodo.org/records/7763972

Initial validation pair:

| Input | Rows | Columns | Stored entries | CSR entries | Explicit zeros |
|---|---:|---:|---:|---:|---:|
| mesh_2d_h005 | 187877 | 187877 | 1311941 | 1311941 | 9692 |
| mesh_2d_h005_P | 187877 | 23081 | 492773 | 492773 | 0 |

Both inputs passed validation and are dimensionally compatible for A * P. Their serial CPU/GPU numerical baselines were subsequently validated below.

## Input Policy

- Supported: Matrix Market coordinate real general/symmetric.
- Values are initially parsed as FP64.
- Convert indices from 1-based to 0-based.
- Sort coordinates by row, then column.
- Reject duplicate coordinates.
- Retain explicit zeros.
- Use 64-bit CSR offsets and 32-bit unsigned indices.

## Implementation Notes

The original GPU prototype still uses dense output and must not be used directly for large datasets. The precision-order program is a controlled demonstration, not an observation of parallel atomic update order.

## Remaining server validation

1. Compile and run the integrated CUDA/atomic/trace tests.
2. Run the integrated experiment on h005, then h001 with bounded references.
3. Review error patterns and diagnostic traces before drawing research conclusions.

## 2026-10-21

- Added structural workload analysis and sparse output construction.
- Added CPU FP32/FP64 numerical baselines using CSR traversal order.
- Added a GPU baseline using identical typed inputs and per-row traversal.
- Controlled CPU/GPU tests passed.
- Reorganized reusable implementations, applications, tests, and examples.
- Added a Makefile with centralized floating-point compilation flags.
- Rebuilt all programs and reran module tests after reorganization.

### Initial dataset results

For mesh_2d_h005 multiplied by its companion P:

- Structural products: 3,471,186.
- Maximum products per output row: 31.
- Structural output entries: 960,341.
- Maximum products per output entry: 10.
- Single-product output entries: 190,587.
- CPU zero-valued output entries: 3,400 in both precisions.
- CPU/GPU bitwise differences: 0 in FP32 and FP64.
- Non-finite output pairs: 0 in both precisions.
- GPU: Quadro RTX 6000, compute capability 7.5.

Agreement is established for this dataset and configuration.
High-precision reference errors and parallel accumulation remain untested.

## 2026-10-22 integrated implementation

- Added a single experiment entry point with CPU-only and CUDA builds.
- Added precision-specific exact SpGEMM references, deterministic sampling/full mode, and multiplication/accumulation error decomposition.
- Added CPU compare-and-exchange accumulation and GPU native atomic accumulation.
- Added separately labeled full-execution diagnostic traces with replay checks.
- Added complete-output repeat summaries, selected-entry ranges, input/product/column statistics, and phase timings.
- Added a wrapper recording hashes, Git state, environment, status, and wall time.
- Added portable build rules and a pinned Boost setup script.
- Local CPU/reference tests and the independent Python Fraction integration oracle passed. The integration test also checks error decomposition, sampling, trace replay, repeat summaries, provenance, overwrite rejection, and failed-run status.
- CUDA compilation/execution and full UIUC dataset runs remain pending on the server; this Mac has no CUDA compiler or NVIDIA GPU.

Previously measured h001 symbolic results: 24,783,775 structural entries, 89,802,248 products, at most 11 products per output entry, 62.88 seconds process wall time and 1,017,948 KiB peak resident memory on the server. These are historical measurements, not results from the new integrated runner.

### Local validation

- `make all test-cpu`: passed with Clang on macOS.
- UndefinedBehaviorSanitizer run of the integrated CPU/reference tests: passed.
- AddressSanitizer could not initialize in this macOS execution environment; no ASan result is claimed.
- Controlled order/product case completed through the manifest wrapper; traces replayed successfully. One observed FP32 atomic run varied across repeats, while variability is not required for a pass.
- A generated sparse stress case with 474,369 structural output entries completed using 1,000 reference entries per precision. This is a synthetic smoke test, not a UIUC dataset result or a server performance estimate.
- CUDA-enabled host orchestration compiled locally and CUDA build commands were dry-run checked. CUDA kernels and the CUDA integration oracle remain uncompiled/unexecuted locally.
