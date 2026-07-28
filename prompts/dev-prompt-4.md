# Development Prompt 4

## User Request
Rewrite `GemmSharedFP32` from scratch while keeping it as a `Gemm` subclass.

The implementation must override `execute`, so it should not be limited by the old `Gemm` dispatch/tile scheme.

## What To Implement
- Redesign `GemmSharedFP32` as its own shared-memory GEMM operator.
- Override `execute` directly and use a dispatch layout that matches the new shared-memory/register-blocked shader.
- Replace the current one-output-per-thread shared shader with a register-blocked tiled shader.
- Keep the public call signature compatible with `GemmArguments`.
- Keep `GemmSharedFP32` inheriting from `Gemm`.
- Update the performance test tile defaults to match the redesigned kernel.

## How To Implement
- Add a base `Gemm` construction path that lets subclasses override `execute` without creating the default GEMM pipeline.
- Use a larger C block size with multiple accumulators per invocation.
- Use workgroup shared memory for A/B tiles and barriers around shared tile reuse.
- Use an initial register-blocked layout with `block_m=64`, `block_n=16`, `block_p=64`, and 4x4 C accumulators per invocation.
- Preserve existing file naming and FP32 postfix conventions.
- Do not add external dependencies.
- Do not compile or execute the project in the agent environment.
