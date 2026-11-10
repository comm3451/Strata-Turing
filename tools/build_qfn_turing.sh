#!/usr/bin/env bash
# Build with the project-local toolchain prepared by bootstrap_turing_toolchain.sh.
set -euo pipefail
strata_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${STRATA_TURING_ISOLATED:-}" != 1 ]]; then
    exec bash "$strata_root/tools/turing-run.sh" bash "$strata_root/tools/build_qfn_turing.sh" "$@"
fi
cd "$strata_root"
strata_tc="$strata_root/.turing/toolchain"
export PATH="$strata_tc/bin:$PATH"
cmake --preset qfn-turing \
    -DCMAKE_C_COMPILER="$strata_tc/bin/x86_64-conda-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="$strata_tc/bin/x86_64-conda-linux-gnu-g++" \
    -DCMAKE_CUDA_COMPILER="$strata_tc/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$strata_tc/bin/x86_64-conda-linux-gnu-g++" \
    -DCUDAToolkit_ROOT="$strata_tc" -DSTRATA_BUILD_TESTS=ON
if (( $# == 0 )); then
    set -- strata strata_kq prefill_fused_iq_test
fi
cmake --build build/qfn-turing --parallel 4 --target "$@"
