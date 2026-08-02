# Remove obsolete GEMM implementations

## What to implement

- Remove the entire GreedyRegister series, including GEMM in-place,
  GEMM out-of-place, and MatMul implementations.
- Remove the current contiguous GEMM implementation (`GemmContiguousNaiveFP32`) so it can be redesigned from scratch later.
- Remove `GemmSharedFP32` so it can be redesigned from scratch later.
- Remove their tests and performance-test modes.

## How to implement

- Delete the corresponding public headers, C++ implementations, and compute shaders.
- Remove the deleted sources and shaders from CMake source lists.
- Remove contiguous/shared/in-place-greedy modes, helpers, includes, and dispatch branches from performance tests.
- Remove correctness tests and sweep scripts that only cover the deleted implementations.
- Remove contiguous-only argument types and flags after verifying that no remaining implementation uses them.
- Remove all GreedyRegister headers, sources, shaders, performance modes,
  and references so only Naive matrix-multiplication implementations remain.
- Do not compile or execute the project in the agent environment.
