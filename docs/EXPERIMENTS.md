# Experiment definitions

## Common problem

Compute `C = A * P`. Inputs are real coordinate Matrix Market matrices. The reader converts indices to zero-based, expands symmetric storage, sorts by row and column, rejects duplicate coordinates, and retains explicit zeros. Input text is initially parsed into binary64. Pattern/integer/complex Matrix Market fields are not currently supported.

A structural output entry exists whenever at least one stored input pair contributes to its coordinate, including products involving explicit zeros. Numerical zeros remain in the output. Output columns are sorted; this storage ordering does not reorder the serial product traversal.

For each precision, the same converted value arrays are used by all implementations. The reference is specific to those typed inputs. It does not claim to recover the exact real numbers intended by decimal input text. `conversion.csv` separates conversion changes from subsequent arithmetic errors.

## Strategies

| Backend | Serial | Atomic |
|---|---|---|
| CPU | One thread, A-row entries then matching P-row entries | Host workers cyclically own A-entry positions; each product updates a shared output using `std::atomic<T>` compare-and-exchange |
| GPU | One thread owns an output row at a time, using identical A/P traversal | One block owns an output row at a time; block threads cyclically own A-row entries and update output using native `atomicAdd` |

Both GPU variants locate output columns by binary search in the same sparse pattern. GPU buffers are allocated, uploaded, initialized, and downloaded on every invocation. A CPU serial baseline is computed once per precision even for GPU-only runs, and its time is reported separately. Host serial and atomic implementations use different workspaces but initialize each output sum to positive zero. Products are separately rounded before addition; FMA contraction and fast math are disabled.

Atomic update order is unspecified, not assumed random. Identical repeated outputs are valid observations. These algorithms are readable research baselines, not optimized-library performance competitors. CPU/GPU atomic strategies do not impose the same order, so differences between them cannot be attributed solely to device hardware.

Native global-memory FP32 CUDA atomics can flush subnormal operands/results even with `--ftz=false`; this is a primitive-level arithmetic distinction, not evidence of an order difference. Inspect the input/product magnitude fields and GPU probe output before attributing differences to scheduling. CPU floating-point environments and compiler behavior also matter. The runner requires host round-to-nearest and records conversion-created zeros/subnormal inputs.

## Exact references

Finite binary32 and binary64 inputs are exact dyadic numbers `s * 2^e`. Their integer significands are multiplied with arbitrary-precision integers; sums align binary exponents without rounding. Non-finite inputs are rejected by the reader. The reference can represent exact products below binary64 underflow and above binary64 overflow.

For each selected entry:

- `R` is the exact sum of exact products of the typed inputs.
- `Q` is the exact sum of products individually rounded in the host target precision.
- `Y` is the observed output of one implementation.
- Signed total arithmetic error: `Y - R`.
- Product error: `Q - R`.
- Accumulation error: `Y - Q`.

The last two sum exactly to the total error when `Q` and `Y` are finite. The accumulation term includes addition-primitive behavior such as flushing, not only order effects. If a host rounded product is non-finite, `Q` is marked invalid and the decomposition is omitted. Non-finite observed outputs are counted separately, not assigned a finite error.

Maximum absolute error and the selection of maximum relative error are compared using exact integer arithmetic. Decimal presentation/division uses Boost's 100-decimal-digit type; printed values are approximations. No ordinary FP64 or long-double result serves as ground truth.

Reference records also include path count, product signs, exact-product magnitude range, sum of absolute products, cancellation condition `sum(abs(products)) / abs(R)`, A-row length, P-column length, and output row/column lengths. Explicit-zero products count toward path length. Minimum nonzero product magnitude is absent when all products are zero.

## Sampling and memory

The same deterministic output-slot sample is used across precisions, backends, strategies, and repeats because they share one output pattern. Sampling is evenly spaced in CSR slot order, not random or stratified; it can miss rare error patterns. It supports controlled comparisons but cannot establish whole-matrix error maxima or rates. Use full mode to make whole-output claims.

With sampled references, exact work is restricted to rows containing selected entries and to selected columns within those rows. No complete list of intermediate products is materialized. Exact objects are retained only for selected entries. The ordinary output still contains every structural entry.

The runner retains a CPU serial baseline and, for the active strategy, first/minimum/maximum output arrays for repeated-run summaries. References and input statistics also need memory. This is sparse storage proportional to input/output sizes plus a per-column workspace, not a dense C allocation. Full references for the largest pair can consume many GB and generate very large CSV files. Start with `--samples 0` for timing, then a bounded sample.

## Trace semantics

`--trace ROW,COLUMN` performs an additional full atomic execution per selected backend and precision. It is not enabled during ordinary repetitions.

Every update to that one output slot uses a lock around the native atomic operation and its event record. The record contains actual A/P source positions, worker ID, rounded product, and before/after values with raw bits. Other slots continue to use ordinary atomics. Source uniqueness, path count, and final output are checked.

CPU traces are replayed with host target-precision addition, checking every before/after value. GPU traces are replayed by a one-thread GPU kernel using native global-memory atomics, preserving that primitive's subnormal behavior. Trace locks require independent thread scheduling; the GPU implementation requires compute capability >= 7.0 for tracing and is targeted at sm_75.

A successfully replayed trace explains the instrumented execution only. Locking and logging perturb scheduling and performance. It does not reveal or prove the uninstrumented execution's order. Do not infer native atomic order from event tickets acquired outside the update, timestamps, thread IDs, or matching final values. `atomic_trace` rows are explicitly separated from ordinary `atomic` rows.

## Timings and provenance

CPU `compute` is host wall time; serial includes its validation/workspace work, and atomic includes worker launch/join. GPU `compute` uses CUDA events around the kernel. GPU `prepare` includes validation, allocation, upload, and initialization; `download` includes result/trace transfer. `call_wall` is measured outside the backend call and includes resource destruction. These scopes differ and are named rather than treated as equivalent kernel throughput measurements.

The wrapper records source-file SHA-256 hashes in addition to Git state, including uncommitted source changes. Commit a clean checkpoint before collecting final research results. CPU thread affinity is not pinned.

The wrapper records end-to-end process wall time separately from input hashing, which warms the OS file cache. Neither value is a cold-cache guarantee. Reference generation happens once per precision, not on every repeated numerical run. Reports are flushed after each run; `status.txt` and the wrapper manifest distinguish failed/interrupted runs from completed ones. Failed/partial reports must not be used as completed experiments.

For a comparison, hold inputs, precision, compilation flags, GPU, thread settings, sampling, and zero/duplicate policies fixed; vary one strategy/backend factor at a time. Record repeated-run variation as supplementary evidence. Device-specific atomic ordering and arithmetic distinctions must be acknowledged when interpreting cross-device comparisons.
