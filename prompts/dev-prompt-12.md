# Development Prompt 12

## What to implement

Complete the unfinished parts of `shaders/MatMulNaiveFP32.new.comp` as an FP32 tiled matrix-multiplication compute shader specialized for subgroup size 32.

After validating that implementation structure, replace the existing
`MatMulNaiveFP32`, `GemmNaiveFP32`, and `GemmOutPlaceNaiveFP32` shader
implementations with variants based on the new shared-memory/workgroup/subgroup
tiling scheme. Update their C++ constructors, specialization setup, dispatch
sizes, or execute paths where required to match the new shader contract.

Expose the two updated naive GEMM paths through the performance-test command
line:

- `--naive` selects/executes the in-place `GemmNaiveFP32` benchmark path.
- `--naive-oop` selects/executes the out-of-place
  `GemmOutPlaceNaiveFP32` benchmark path.

- A workgroup contains a specialization-controlled grid of subgroups whose
  dimensions are `subgroup_tile_cnt_m` by `subgroup_tile_cnt_p`.
- Each subgroup owns a distinct output C tile and keeps its C accumulators in registers.
- A and B are loaded cooperatively by the entire workgroup into a larger shared-memory tile.
- Matrix bounds and the existing arbitrary tensor strides must be handled correctly.
- The completed accumulator values are written to C after the entire N reduction.

## How to implement

- Treat `gl_WorkGroupID.x`, `.y`, and `.z` as batch, workgroup-M-tile, and workgroup-P-tile indices.
- Cooperatively load the full A and B shared tiles using `gl_LocalInvocationIndex`, independently of subgroup ownership, and zero-fill out-of-bounds elements.
- Synchronize the workgroup after the shared loads and again before the next N tile overwrites shared memory.
- Map `gl_SubgroupID` to the specialization-controlled 2D subgroup grid and
  `gl_SubgroupInvocationID` to the subgroup's M/P lane grid; do not hardcode
  the subgroup count.
- Within each subgroup, read A and B fragments from shared memory and use subgroup shuffles to distribute them to the lanes that own distinct register C micro-tiles.
- Initialize the register accumulator once, accumulate across every shared N tile, and perform one bounds-checked writeback at the end.
- Keep `main()` minimal. Separate index calculations, cooperative shared-memory
  loads, register initialization, shared-to-register fragment loads, subgroup
  accumulation, synchronization phases, and writeback into small helper
  functions wherever GLSL permits so that each stage can be debugged
  independently.
- Add a leading shader comment that documents the complete dispatch hierarchy:
  workgroup IDs, the specialization-controlled subgroup grid, subgroup IDs,
  subgroup invocation IDs, lane M/P coordinates, register-tile ownership, and
  the shared-N iteration/barrier sequence.
- Further separate shared/global coordinate decoding and lane/subgroup output
  offsets into small scalar index helpers. Keep the innermost FMA loop direct
  when additional helper boundaries could interfere with optimization.
- Do not add external libraries and do not compile or execute the shader in the agent environment.
- Preserve each operation's semantics:
  - `MatMulNaive`: `C = A * B`
  - `GemmNaive`: in-place `C = alpha * A * B + beta * C`
  - `GemmOutPlaceNaive`: `D = alpha * A * B + beta * C`
- Reuse the debug-friendly helper structure and hierarchy documentation from
  `MatMulNaiveFP32.new.comp` while adapting buffer bindings and epilogues for
  each operation.
- Follow the existing `PerformanceTest` option parsing, reporting, and benchmark
  conventions. Keep the two new options independently selectable and preserve
  existing benchmark modes.

Split the subgroup/shared-memory kernels into two named strategies:

- Rename the current A/B register-fragment implementation and its C++ wrappers
  from `Naive` to `GreedyRegister` for MatMul, in-place GEMM, and out-of-place
  GEMM. Preserve its behavior and `reg_tile_n` preloading strategy.
- Recreate the `Naive` shader/C++ names with a lower-register-pressure streaming
  implementation:
  - keep only the persistent C accumulator tile plus the current A/B values;
  - for each shared-N position, let designated subgroup lanes load the needed
    shared values;
  - immediately distribute them with subgroup shuffles and perform the FMA;
  - do not retain A/B fragments across multiple N positions.
- Keep specialization ID 9 and the shared-N depth formula compatible across
  both strategies. Interpret it as `reg_tile_n` for GreedyRegister and
  `k_unroll` for Naive.
- Add both strategies to shader/source CMake lists and expose GreedyRegister
  benchmark modes alongside the existing `--naive` and `--naive-oop` streaming
  modes.
