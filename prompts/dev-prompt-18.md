# Register-tiled naive reduction

## What to implement

- Optimize the naive FP32 reduction operators (`Sum`, `Avg`, `Min`, and `Max`)
  so each shader invocation holds and reduces a register tile of values, as in
  the optimized GEMM kernel.
- Make the number of values held by each invocation configurable, targeting
  32 to 64 values per thread.
- Update the reduction specialization constants and their C++ bindings to
  include the per-thread register-tile size.
- Preserve the current public reduction behavior, including strided input,
  batching, deterministic first-index tie breaking for `Min`/`Max`, and the
  existing empty-reduction outputs.

## How to implement

- Add a `values_per_thread` specialization constant to all four naive
  reduction shaders, with a default of 64.
- Load up to `values_per_thread` input elements into per-invocation register
  accumulators, handling the final partial tile with bounds checks, then reduce
  those accumulators before the existing subgroup and workgroup reduction.
- For `Min` and `Max`, retain the matching global index alongside each register
  value and preserve the lowest-index result when values are equal.
- Add `values_per_thread` to each naive reduction operator constructor and pass
  it to the shader pipeline as specialization constant ID 1. Keep existing
  call sites source-compatible by giving it a default value of 64.
- Make only the minimal changes needed in the four shaders, the naive reduction
  header/source, and focused correctness tests if existing coverage does not
  exercise partial register tiles and index tie breaking.
- Do not add external libraries.
- Do not compile or execute the project in the agent environment.
