#!/bin/bash
#SBATCH --job-name=ml-jet-train                # Job name
#SBATCH --partition=ecscp                     # Partition (adjust if needed)
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:2  # 2 A100 GPUs
#SBATCH --cpus-per-task=8                     # Number of CPU cores
#SBATCH --mem=8G                            # Memory allocation
#SBATCH --time=12:00:00                       # Maximum run time (12 hours)
#SBATCH --nodes=1                             # Single node
#SBATCH --ntasks=1                            # Single task
#SBATCH --qos=express                         # Queue (express for quicker scheduling)

# -------------------------------
# 1. Model Arguments (Edit These for Each Run)
# -------------------------------
MODEL_TAG="EfficientNet"
BATCH_SIZE=512
EPOCHS=50
LEARNING_RATE=0.001
GLOBAL_MAX=121.79151153564453
ROOT_DIR=~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled
OUTPUT_DIR=training_output/

# -------------------------------
# 2. Dynamic Log File Naming (Shell-handled)
# -------------------------------
RUN_TAG="${MODEL_TAG}_bs${BATCH_SIZE}_ep${EPOCHS}_lr$(printf %.0e $LEARNING_RATE)"
LOGS_DIR="logs"

mkdir -p $LOGS_DIR

LOG_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.out"
ERR_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.err"

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Error file: $ERR_FILE"

# Redirect stdout and stderr manually (since SLURM directives cannot use variables)
exec > $LOG_FILE 2> $ERR_FILE

# -------------------------------
# 3. Environment Setup
# -------------------------------
echo "[INFO] SLURM Job started on: $(date)"
module purge
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh
conda init
conda activate tensorflow-gpu-v2.8

# -------------------------------
# 4. Record Start Time for Training Timer
# -------------------------------
START_TIME=$(date +%s)

# -------------------------------
# 5. Run Training Script
# -------------------------------
python train.py \
--root_dir $ROOT_DIR \
--global_max $GLOBAL_MAX \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--learning_rate $LEARNING_RATE \
--model_tag $MODEL_TAG \
--output_dir $OUTPUT_DIR

# -------------------------------
# 6. Record End Time and Compute Duration
# -------------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "[INFO] Training completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"

# -------------------------------
# 7. End of Job
# -------------------------------
echo "[INFO] SLURM Job finished on: $(date)"
