# Add serial GEMM measurement and warm-up

## What to implement

- Change `PerformanceTest` so measured GEMM iterations execute serially rather
  than being recorded together and submitted as one batch.
- Add warm-up GEMM iterations before timed measurement.
- Exclude warm-up work from the reported elapsed time and throughput.

## How to implement

- Create the operator's `DispatchPlan` and one-dispatch `ExecutionPlan` once,
  outside the warm-up and timed loops.
- Reuse that same `ExecutionPlan` for every warm-up and measured iteration by
  calling `execute()` and waiting for the returned token before starting the
  next iteration.
- Keep buffer allocation and operator construction outside the timed loop.
- Use an explicit warm-up iteration count and report it with the benchmark
  configuration.
- Preserve the existing command-line interface and measured-iteration count
  unless the benchmark already exposes an appropriate warm-up option.
- Modify only the benchmark source needed for this behavior.
- Do not compile or execute the project in the agent environment.
