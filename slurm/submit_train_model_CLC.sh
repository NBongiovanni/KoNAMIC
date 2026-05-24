#!/bin/bash

set -e

GPU_TYPE="$1"
shift

case "${GPU_TYPE}" in
  full)
    sbatch \
      --mem=120G \
      --cpus-per-task=12 \
      --gpus=full \
      --time=12:00:00 \
      train_model_CLC.sh "$@"
    ;;

  slice)
    sbatch \
      --mem=44G \
      --cpus-per-task=6 \
      --gpus=slice \
      --time=15:00:00 \
      train_model_CLC.sh "$@"
    ;;

  *)
    echo "Usage: $0 {full|slice} [training arguments...]"
    exit 1
    ;;
esac