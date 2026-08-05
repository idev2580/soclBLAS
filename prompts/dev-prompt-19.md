# Register-tiled naive dot product

## What to implement

- Optimize the naive FP32 dot product so each shader invocation multiplies and
  accumulates multiple input pairs instead of processing one scalar pair.
- Follow the register-tiled approach already used by the naive reduction
  operators, targeting 32 to 64 values per thread.
- Preserve the current public dot-product behavior, including strided inputs,
  batching, and correct handling of sizes that do not fill a complete tile.
- Add reusable unary and binary elementwise operators whose GLSL operation can
  be supplied independently from a shared full-shader template.
- Unary operations define `float operation(float x)`, while binary operations
  define `float operation(float x, float y)`.
- Elementwise operators are always out of place and process every value in the
  supplied buffer size without tensor stride arguments.

## How to implement

- Compare `DotProductNaive` with the existing register-tiled reduction shaders
  and C++ specialization-constant bindings.
- Add a configurable `values_per_thread` specialization constant to the dot
  product shader, defaulting to 64 for source compatibility and consistency
  with reductions.
- Have each invocation accumulate up to `values_per_thread` products in
  registers, bounds-checking the final partial tile before the existing
  subgroup/workgroup reduction.
- Update the C++ shader specialization bindings while retaining the existing
  one-workgroup-per-batch dispatch model.
- Add or adjust focused correctness tests if current coverage does not exercise
  partial tiles, strides, and batches under the new dispatch scheme.
- Follow the shader-template and injected-function pattern used by the GEMM
  epilogue implementation, keeping common unary and binary indexing, bounds
  checks, and output writes in shared shader templates.
- Expose C++ unary and binary elementwise operator APIs that accept buffer size
  rather than tensor dimensions or strides, bind separate output buffers, and
  dispatch enough invocations to cover that size.
- Add focused tests with simple custom unary and binary GLSL operations,
  including a size that is not divisible by the workgroup size.
- Do not add external libraries.
- Build the project and fix build errors after implementation. Do not run the
  resulting tests or executables unless separately authorized.
