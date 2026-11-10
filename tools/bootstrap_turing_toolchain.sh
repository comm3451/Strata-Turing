#!/usr/bin/env bash
# NVIDIA CUDA 13.0 + GCC 14, wholly inside this checkout; no shell activation.
set -euo pipefail
strata_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${STRATA_TURING_ISOLATED:-}" != 1 ]]; then
    exec bash "$strata_root/tools/turing-run.sh" bash "$strata_root/tools/bootstrap_turing_toolchain.sh"
fi
cd "$strata_root"
mkdir -p .turing/downloads .turing/mamba-bootstrap
strata_archive="$strata_root/.turing/downloads/micromamba.tar.bz2"
if [[ ! -f "$strata_archive" ]]; then
    curl --fail --location --retry 3 \
        https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2 -o "$strata_archive"
fi
echo "9a97976244929bbae28fb1fae504e74a3e7e6d1932870e3a03321f01b7fbd07b  $strata_archive" | sha256sum --check -
python3 - "$strata_archive" <<'PY'
import pathlib, sys, tarfile
with tarfile.open(sys.argv[1], "r:bz2") as archive:
    member = archive.extractfile("bin/micromamba")
    dest = pathlib.Path(".turing/mamba-bootstrap/bin/micromamba")
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(member.read())
    dest.chmod(0o755)
PY
export MAMBA_ROOT_PREFIX="$strata_root/.turing/mamba"
export CONDA_PKGS_DIRS="$strata_root/.turing/cache/conda"
# `create` registers in ~/.conda/environments.txt. Install into an existing empty
# prefix instead: micromamba recognizes conda-meta/history without registration.
mkdir -p .turing/toolchain/conda-meta
touch .turing/toolchain/conda-meta/history
.turing/mamba-bootstrap/bin/micromamba --no-rc install --yes \
    --prefix "$strata_root/.turing/toolchain" \
    --file tools/turing-toolchain-linux-64.txt
