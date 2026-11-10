#!/usr/bin/env bash
# Run an experiment with ordinary files outside this checkout mounted read-only.
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: bash tools/turing-run.sh COMMAND [ARG ...]" >&2
    exit 2
fi
command -v bwrap >/dev/null || {
    echo "turing-run: bubblewrap (bwrap) is required; no command was run." >&2
    exit 1
}

strata_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
strata_state="$strata_root/.turing"
# Refuse a redirected state folder before creating anything.
if [[ -L "$strata_state" ]]; then
    echo "turing-run: .turing must be a real directory inside the checkout." >&2
    exit 1
fi
mkdir -p -- "$strata_state"

# Expose GPU character devices, but keep the host's /dev/shm out of the sandbox.
strata_devices=()
for strata_device in /dev/nvidia* /dev/nvidia-caps/*; do
    if [[ -c "$strata_device" ]]; then
        strata_devices+=(--dev-bind "$strata_device" "$strata_device")
    fi
done

# Make directories after mounting: even an existing nested symlink cannot cause a write
# outside the checkout. HOME stays unchanged so the existing install remains readable.
exec bwrap --die-with-parent --unshare-pid \
    --ro-bind / / --bind "$strata_root" "$strata_root" \
    --dev /dev "${strata_devices[@]}" --proc /proc --chdir "$strata_root" \
    --setenv STRATA_TURING_ISOLATED 1 \
    --setenv TMPDIR "$strata_state/tmp" \
    --setenv TMP "$strata_state/tmp" --setenv TEMP "$strata_state/tmp" \
    --setenv XDG_CACHE_HOME "$strata_state/cache" \
    --setenv XDG_CONFIG_HOME "$strata_state/config" \
    --setenv XDG_DATA_HOME "$strata_state/data" \
    --setenv XDG_STATE_HOME "$strata_state/state" \
    --setenv CUDA_CACHE_PATH "$strata_state/cache/cuda" \
    --setenv HF_HOME "$strata_state/cache/huggingface" \
    --setenv PIP_CACHE_DIR "$strata_state/cache/pip" \
    --setenv UV_CACHE_DIR "$strata_state/cache/uv" \
    --setenv CCACHE_DIR "$strata_state/cache/ccache" \
    --setenv STRATA_DATA_DIR "$strata_state/data/strata" \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    -- /bin/bash -c '
        set -e
        mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
            "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$CUDA_CACHE_PATH" \
            "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$CCACHE_DIR" "$STRATA_DATA_DIR"
        exec "$@"
    ' turing-run "$@"
