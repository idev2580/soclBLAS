# Correct GemvNaive GEMM construction

## What to implement

- Correct the `GemvNaiveFP32` construction path so that it uses the `GemmNaiveFP32` configuration and constructor semantics.
- Initialize the generic `Gemv` layer from the context-only `Gemm` constructor where appropriate.
- Remove the accidental mapping of Gemv's `tile_k` argument to GEMM's `subgroup_tile_p` argument.
- Use the measured best configuration `1 1 32 4 2 4 16 1 4` for fixed
  Naive GEMM, Gemv, MatMul, and out-of-place constructor calls in tests.

## How to implement

- First reconcile the current hierarchy, where `Gemv` contains a `Gemm` member rather than inheriting from `Gemm`, with the requested inheritance-based construction.
- Preserve the GEMV-to-GEMM argument conversion behavior.
- Leave sweep configurations and command-line performance configurations
  configurable rather than replacing them with a fixed test configuration.
- Make the smallest hierarchy and constructor changes needed after the intended ownership/inheritance model is confirmed.
- Do not compile or execute the project in the agent environment.
