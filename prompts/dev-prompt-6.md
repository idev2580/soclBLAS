# Prompt 6: Vectorized contiguous GEMM naive FP32

## User Prompt Summary

The current work is developing GEMM under a contiguous memory assumption. The problem setting has been relaxed, so the implementation should now use vectorized loads and vectorized calculation where appropriate.

## What to Implement

- Implement the `GemmContiguousNaiveFP32.comp` shader.
- Add `GemmContiguousArguments` to `GemmArguments.hpp`.
- Add a C++ `GemmContiguousNaiveFP32` operator implementation and header.
- Make `GemmContiguousNaiveFP32` inherit `Operator` directly, not `Gemm`, because the push constant ABI differs from the generic GEMM path.

## How to Implement

- Follow the shared-memory GEMM structure for tiling and value reuse.
- Do not use the non-shared GEMM/MatMul shaders as implementation references because their value reuse characteristics are different.
- Keep the implementation scoped to the contiguous GEMM path.
- Use vectorized shader operations for FP32 contiguous matrix data while preserving correct GEMM semantics.
- Use a contiguous-specific push constant layout: batch/m/n/p, alpha/beta, matrix-wise strides, major column-wise strides, and flags. Do not use generic row/column element strides in the contiguous shader path.
- Keep `operator()(A, B, C, args)` similar to the original GEMM call shape, but accept `GemmContiguousArguments`.
- Do not implement scalar-load fallback paths in the contiguous shader. The contiguous implementation is for an aligned/padded relaxed problem setting, so unsupported dimensions/layouts should be rejected on the host side instead of handled with per-element conditionals in the shader.
- Remove helper logic such as component selection or ternary scalar fallback that defeats the point of vectorized contiguous access.
- Require vector-friendly tile configuration and dimensions on the host side: block sizes, per-thread tile sizes, dimensions, and strides must satisfy the shader's vec4 assumptions.
- Keep the contiguous shader centered on the normal row-major GEMM dot-product path. Do not switch the compute structure into a row-vector/output-transposed accumulator path just because the output storage is transposed.
- Final C update must use vectorized C load, alpha/beta calculation, and vectorized C store. For transposed C, build the output vec4 from the already computed accumulator lanes according to the transposed storage layout.
- Keep the shader debuggable: factor A/B shared tile loading into separate functions, keep the dot-product accumulation section clearly commented, and keep the C update/writeback section clearly commented.
- Update `PerformanceTest.cpp` so the existing shared GEMM performance flow is split into `test_gemm()`, add `test_cont_gemm()` for `GemmContiguousNaiveFP32`, and select between them with a command-line option.
- Do not add new external libraries.
- Do not compile or run tests in the agent environment; leave build and execution validation to the developer.
