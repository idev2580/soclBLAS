# Development Prompt 5

## User Request
Make the per-thread output tile size configurable for `GemmSharedFP32`.

The implementation already controls block tile sizes with specialization constants, so thread output tile dimensions should be controlled the same way.

## What To Implement
- Add configurable per-thread output tile dimensions to `GemmSharedFP32`.
- Pass thread tile dimensions as specialization constants.
- Compute local workgroup size from `block_m / thread_tile_m` and `block_p / thread_tile_p`.
- Update the performance test to allow command-line overrides for thread tile dimensions.

## How To Implement
- Keep `GemmSharedFP32` inheriting from `Gemm` and overriding `execute`.
- Use a bounded accumulator array in the shader because GLSL array sizes must be compile-time constants.
- Validate that block dimensions are divisible by thread tile dimensions.
- Keep the default conservative but more useful than 2x2, starting with a 4x4 per-thread output tile.
- Do not compile or execute the project in the agent environment.
