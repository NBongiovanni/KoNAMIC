#!/usr/bin/env bash
set -euo pipefail

export ACADOS_INSTALL_DIR="$HOME/acados"
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

# Se placer dans le dossier du script
cd "$(dirname "$0")" || exit 1

# Ajouter la racine (parent de slurm) au PYTHONPATH
export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"

python -m entrypoints.run_closed_loop_simulation "$@"