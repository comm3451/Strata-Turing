#!/usr/bin/env bash
# Serve the node's private, measured profile with checkout-local writable paths.
set -euo pipefail

strata_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
strata_python="$HOME/.local/share/strata/terminal/0.1.56/python/bin/python3"
strata_server="$HOME/.local/share/strata/components/control/0.1.57/serve/server.py"
strata_config="$strata_root/.turing/launch/qfn-iq3s-256k.json"

[[ -x "$strata_python" && -r "$strata_server" && -r "$strata_config" ]] || {
    echo 'turing-serve: the measured Python/server installation and private QFN profile are required.' >&2
    exit 1
}

exec bash "$strata_root/tools/turing-run.sh" \
    "$strata_python" "$strata_server" \
    --engine strata --config "$strata_config" --port 8080
