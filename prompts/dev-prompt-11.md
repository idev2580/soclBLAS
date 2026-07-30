# GemmContiguous N-dimension tile-load offset optimization

## What to implement

- Investigate whether `GemmContiguous` can load tiles along the N dimension by incrementing a precomputed offset instead of evaluating a multiplication for each element.
- Evaluate whether power-of-two tile dimensions should use explicit bit shifts, taking specialization-constant optimization into account.
- Identify the highest-value memory-bandwidth optimizations in the existing global-load, shared-memory, and C-store paths.
- Make thread-item tiles and block-tile configurations more flexible, including rectangular per-thread shapes such as `16x4`.
- Preserve every supported A/B/C transpose layout and the existing contiguous/aligned access assumptions.
- If implementation is requested after the design review, optimize the relevant shader indexing without adding external libraries.

## How to implement

- Inspect the current `GemmContiguousNaiveFP32` shader and its host-side configuration/argument layout.
- Identify which N-direction address expressions contain loop-varying multiplication and distinguish compile-time/uniform work from per-invocation work.
- Distinguish multiplication or division by power-of-two specialization constants, which the shader compiler can normally lower to shifts, from multiplication by runtime major strides.
- Hoist invariant base/stride calculations outside loops, then use an offset recurrence (`offset += stride`) where it reduces dynamic integer arithmetic without harming vectorized/coalesced loads.
- Account explicitly for transposed and non-transposed layouts, since the increment is layout-dependent.
- Evaluate global-memory coalescing for every A/B transpose combination, tile reuse/arithmetic intensity, shared-memory bank behavior, and whether double buffering can overlap tile loads with computation.
- Replace the artificial per-axis `thread_tile <= 8` restriction with constraints based primarily on total accumulator count, workgroup size, divisibility, and shared-memory usage.
- Keep subgroup register-count limits separate from per-thread tile-shape limits, and extend correctness/performance configuration coverage for rectangular thread tiles.
- Validate the proposal against existing correctness coverage for all layout flags. Per repository policy, do not compile or execute in the agent environment.

## Developer benchmark observations

- For `M=4096, N=1024, P=1024`, `tile=128x16x128`, `thread_tile=8x8` reached about 1321 GFLOPS and was roughly 5.8% faster than `128x32x128`.
- `tile_n=8` lost most of that gain, indicating a tradeoff between shared-memory residency and increased barrier/load overhead.
- Larger balanced M/P tiles were substantially faster; `128x128` outperformed `64x64` by about 2.47x.
- With equal tile area, `64x128` was materially faster than `128x64`, suggesting the row-oriented subgroup path favors a wider P dimension and fewer subgroup row shuffles.
- The `64x64` thread-tile sweep favored the 128-thread `8x4` shape, while both 256-thread `4x4` and 64-thread `8x8` shapes were slower; register count alone does not explain the result.
- The developer reports DDR5-5600 system memory with an assumed total bandwidth of about 44 GB/s. For the `128x16x128` result, the workgroup-issued logical arithmetic intensity is about 30.1 FLOP/byte, giving a roofline near 1.32 TFLOPS at 44 GB/s, almost exactly matching the measured result.
- Treat the roofline match as strong evidence of a memory-side limit only when the GPU is UMA and 44.8 GB/s is the actual aggregate channel bandwidth; use DRAM/L2 counters to distinguish physical DRAM traffic from cache-served logical tile loads.
- The developer confirmed that `64x16x256` is faster than the symmetric `256x16x64`, supporting the prediction that the non-transposed row-subgroup path benefits from a wider P tile.
- After enabling more flexible thread tiles and up to 128 accumulators, the developer found no material performance gain; `128x16x128` with an `8x8` configured thread tile remains the fastest tested option.
- This is consistent with `128x128 = 256 workgroup threads * 64 accumulators per thread`: the existing winner already uses the largest output tile possible under the 256-thread and 64-accumulator budgets, while larger tiles require 128 accumulators and incur additional register/shared-memory pressure.
- Raising the contiguous shared-memory ceiling to 64 KiB and increasing only the K (`block_n`) tile did not improve performance. This is expected because K-tile size does not increase GEMM arithmetic intensity; it mainly trades fewer barriers for lower shared-memory residency.
- Loading a larger M panel alone cannot provide free reuse: computing multiple output P tiles in the same K pass requires retaining their partial sums, which returns to the same accumulator-register pressure, while processing P tiles sequentially requires reloading A or writing partial C.
- The next architectural experiment should decouple workgroup thread count/local shape from output-tile configuration, allowing 512-thread workgroups on capable devices so a `128x256` output tile can retain 64 accumulators per thread instead of 128.
- Independently consider ping-pong shared tiles with register prefetch of the next K tile; this uses extra shared memory to overlap load latency rather than merely enlarging `block_n`.

## Approved implementation

- Replace the per-axis `thread_tile_m/thread_tile_p <= 8` restriction with a total accumulator limit of 128 so rectangular configurations such as `16x4`, `4x16`, and `8x16` can be explored.
- Remove the unnecessary requirement that thread-tile dimensions themselves be divisible by four; retain the vector-load alignment requirements on block dimensions and strides.
- Keep the subgroup-local temporary-array limits separate from the public thread-tile validation.
- Reorder dispatch/workgroup coordinates to `P tile, M tile, batch` to improve the likelihood of immediate A-tile reuse through cache.
- Extend configuration/correctness coverage for rectangular thread tiles. Do not compile or execute in the agent environment.
- Raise the `GemmContiguous` shared-memory configuration ceiling from 32 KiB to 64 KiB for device-specific performance experiments, while leaving the separate `GemmShared` operator at its existing 32 KiB limit.
- Add a correctness configuration that allocates more than 32 KiB of shared memory so the relaxed contiguous limit is covered on capable devices.
