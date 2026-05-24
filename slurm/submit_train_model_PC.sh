#!/usr/bin/env bash
set -euo pipefail

# Initialiser conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activer l'environnement
conda activate KoNAMIC

export ACADOS_INSTALL_DIR="$HOME/acados"
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

# Se placer dans le dossier du script
cd "$(dirname "$0")" || exit 1

# Ajouter la racine (parent de slurm) au PYTHONPATH
export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"

python -m entrypoints.training.run_train_model "$@"