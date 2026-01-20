#!/bin/bash
#SBATCH --job-name=gpt2_default
#SBATCH --account=<PLACEHOLDER>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=/scratch/%u/gpt2-logs/%x-%j.out
#SBATCH --error=/scratch/%u/gpt2-logs/%x-%j.err


set -euo pipefail

module load StdEnv/2023 gcc arrow

source $HOME/hacker_env/bin/activate

cd $PROJECT_DIR/IFT6164/hacker/

# W&B
export WANDB_PROJECT=gpt2-owt
export WANDB_NAME=gpt2_default

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATA_PATH="$SCRATCH/nanoGPT/data"

test -f "$DATA_PATH/openwebtext/train.bin" || { echo "Missing train.bin under $DATA_PATH/openwebtext"; exit 1; }
test -f "$DATA_PATH/openwebtext/val.bin" || { echo "Missing val.bin under $DATA_PATH/openwebtext"; exit 1; }

python nanoGPT/train.py \
  --dataset=openwebtext \
  --data_path="$DATA_PATH" \
  --out_dir="$SCRATCH/gpt2-experiments/" \
  --wandb_log=True \
  --wandb_project=$WANDB_PROJECT \
  --wandb_run_name=$WANDB_NAME \
  --compile=True