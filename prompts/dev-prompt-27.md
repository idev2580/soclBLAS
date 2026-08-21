# Select deep performance measurements using top-k scan results

## What to implement

- Change `PerformanceSweepTest` so the scan phase selects configurations for
  deep performance measurement by rank instead of a fixed TFLOPS threshold.
- Keep only the k configurations with the highest scan TFLOPS values.
- Preserve the existing scan, error, selected-count, and deep-measurement
  output behavior.

## How to implement

- Replace `SELECTION_THRESHOLD_TFLOPS` with a compile-time top-k count of 10.
- Store each successfully scanned configuration together with its measured
  TFLOPS value.
- Sort successful scan results by descending TFLOPS after the scan completes,
  then select the first `min(k, successful_scan_count)` configurations.
- Preserve all existing uncommitted problem-size changes in
  `tests/PerformanceSweepTest.cpp`.
- Do not add external libraries.
- Do not compile or execute the project in the agent environment.
