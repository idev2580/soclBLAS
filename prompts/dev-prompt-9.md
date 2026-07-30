# Development Prompt 9

## User Request

Inspect `GemmSharedFP32` for shared-memory bank conflicts, determine whether its
compute loop is outer-product based, and optimize it where possible.

The user also asked whether further tuning of the optimized contiguous GEMM now
requires Nsight Compute rather than general optimization heuristics.

## What To Implement

- Analyze the shared-memory address pattern for A and B at subgroup/lane level.
- Confirm the mathematical accumulation structure used by `GemmSharedFP32`.
- Remove or substantially reduce shared-memory bank conflicts in the hot
  accumulation loop.
- Preserve generic strided A/B/C layouts, partial edge tiles, alpha/beta
  semantics, and the existing host-facing interface.

## How To Implement

- Inspect the current shader and host specialization/dispatch geometry.
- Treat the existing per-K register tile as an outer product if it already has
  that structure; do not rewrite it merely to rename the algorithm.
- Store A in a compute-friendly K-major shared layout with a reversible XOR
  bank swizzle for power-of-two M tiles.
- Interleave each invocation's output rows across `local_m_cnt`, so adjacent
  lanes consume adjacent banks while retaining the full register-tile outer
  product. Keep an unswizzled fallback for non-power-of-two M tiles.
- Do not add subgroup operations where shared-memory broadcast already serves
  the same value efficiently.
- Do not add dependencies.
- Request permission before editing files outside `prompts`.
- Do not compile or execute the project in the agent environment.
