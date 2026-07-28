# Development Prompt 3

## User Request
Add an `FP32` postfix to every concrete operation name, following the existing `GemmNaiveFP32` style.

Follow-up naming correction:
- Use the existing CamelCase GEMM naming scheme. The shared-memory GEMM class should be `GemmSharedFP32`.

## What To Implement
- Rename concrete operation classes so FP32 variants consistently end with `FP32`.
- Keep class/file/shader capitalization consistent with existing names such as `GemmNaiveFP32`.
- Keep existing base operation interfaces unchanged where they are generic abstractions.
- Update tests and performance code to use the renamed FP32 classes.
- Preserve existing implementation behavior.

## How To Implement
- Inspect all operation headers and source files to identify concrete FP32 operator classes missing the postfix.
- Do not keep compatibility aliases for the old concrete operation names.
- Update references in tests and performance tests.
- Do not add external dependencies.
- Do not compile or execute the project in the agent environment.
