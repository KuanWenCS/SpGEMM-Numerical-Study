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

Both inputs passed validation and are dimensionally compatible for A * P. Dataset multiplication has not yet been performed.

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

## Next Steps

1. Record input hashes and add basic input/workload statistics.
2. Build sparse output structure.
3. Implement scalable CPU/GPU baselines with matching accumulation order.
4. Compare FP32 and FP64 results under controlled settings.
5. Add high-precision references, parallel accumulation, and validated order diagnostics.

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