# Development Prompt 10

## User Request

Add `PerformanceShaderCorrectnessTest.cpp` to validate the optimized GEMM
shaders against values produced by the existing naive GEMM shaders.

## What To Implement

- Use naive GEMM shader output as the GPU-computed reference result.
- Compare `GemmSharedFP32` and `GemmContiguousNaiveFP32` outputs against that
  reference.
- Cover the optimized performance configuration as well as correctness-relevant
  alpha/beta and layout cases supported by each implementation.
- Integrate the new source into the existing test target.

## How To Implement

- Reuse the context, buffer upload/download, synchronization, and floating-point
  comparison patterns from `NaiveTest.cpp`.
- Initialize identical A, B, and C inputs for the reference and tested kernels.
- Choose dimensions compatible with contiguous-kernel tile restrictions while
  also exercising enough blocks to expose workgroup indexing errors.
- Use tolerance-based comparison and report the first useful mismatch context.
- List gtest sources explicitly instead of globbing, so standalone benchmarks
  cannot be pulled into the test target accidentally.
- Do not add external dependencies.
- Ask permission before changing files outside `prompts`.
- Do not compile or execute the project in the agent environment.
