#!/bin/bash
#SBATCH --job-name=gpt2_stolen_lr_mult
#SBATCH --account=<PLACEHOLDER>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load StdEnv/2023 gcc arrow

source $SCRATCH/hacker_env/bin/activate

cd $PROJECT_DIR/IFT6164/hacker/stealing_model

# W&B
export WANDB_PROJECT=gpt2-owt
export WANDB_NAME=gpt2_stolen_lr_mult

export HF_HOME=$SCRATCH/hf_cache

DATA_PATH="/scratch/salmanhu/nanoGPT/data"
STOLEN_CKPT="$SCRATCH/extraction_results/lm_head_stolen.pt"

test -f "$STOLEN_CKPT" || { echo "Missing stolen ckpt: $STOLEN_CKPT"; exit 1; }
test -f "$DATA_PATH/openwebtext/train.bin" || { echo "Missing train.bin under $DATA_PATH/openwebtext"; exit 1; }
test -f "$DATA_PATH/openwebtext/val.bin" || { echo "Missing val.bin under $DATA_PATH/openwebtext"; exit 1; }

python train.py \
  --dataset=openwebtext \
  --data_path="$DATA_PATH" \
  --stolen_ckpt_path="$STOLEN_CKPT" \
  --emb_lr_mult=0.1 \
  --wandb_log=True \
  --wandb_project=$WANDB_PROJECT \
  --wandb_run_name=$WANDB_NAME \
  --compile=True