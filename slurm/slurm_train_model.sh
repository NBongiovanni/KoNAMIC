#!/bin/bash
#SBATCH --job-name=KSIC
#SBATCH --output=logs/KSIC-%j.stdout
#SBATCH --error=logs/KSIC-%j.stderr

eval "$(conda shell.bash hook)"

conda activate KoNAMIC
which python
python -c "import torch; print(torch.__version__)"

# ---- choix du script Python (vision vs sensor) ----
if [[ "$*" == *"--modality sensor"* ]]; then
  ENTRYPOINT="$HOME/KSIC_v8/entrypoints/run_train_sensor_model.py"
else
  ENTRYPOINT="$HOME/KSIC_v8/entrypoints/run_train_vision_model.py"
fi

echo "Entry point: ${ENTRYPOINT}"
echo "Arguments: $@"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

srun python "${ENTRYPOINT}" "$@"