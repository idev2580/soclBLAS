#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_file="${script_dir}/CublasGemmBenchmark.cpp"
output_file="${1:-${script_dir}/cublas_gemm_benchmark}"
cxx_binary="${CXX:-c++}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "CONDA_PREFIX is not set. Activate the Conda environment containing cuBLAS first." >&2
    exit 1
fi

if ! command -v "${cxx_binary}" >/dev/null 2>&1; then
    echo "C++ compiler '${cxx_binary}' was not found." >&2
    exit 1
fi
cxx_path="$(command -v "${cxx_binary}")"

target_root="${CONDA_PREFIX}/targets/x86_64-linux"
compiler_flags=()

for include_dir in "${target_root}/include" "${CONDA_PREFIX}/include"; do
    if [[ -d "${include_dir}" ]]; then
        compiler_flags+=("-I${include_dir}")
    fi
done

for library_dir in "${target_root}/lib" "${CONDA_PREFIX}/lib"; do
    if [[ -d "${library_dir}" ]]; then
        compiler_flags+=(
            "-L${library_dir}"
            "-Wl,-rpath,${library_dir}"
        )
    fi
done

env \
    -u LD_LIBRARY_PATH \
    "PATH=/usr/bin:/bin:${PATH}" \
    "${cxx_path}" \
    -O3 \
    -std=c++17 \
    "${compiler_flags[@]}" \
    "${source_file}" \
    -lcublas \
    -lcudart \
    -o "${output_file}"

echo "Built ${output_file}"
