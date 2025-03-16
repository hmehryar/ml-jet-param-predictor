#!/bin/bash
#SBATCH --job-name=ml-jet-train-mamba
#SBATCH --partition=ecscp
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=express

MODEL_TAG="Mamba"
BATCH_SIZE=512
EPOCHS=50
LEARNING_RATE=0.001
GLOBAL_MAX=121.79151153564453
ROOT_DIR=~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled
OUTPUT_DIR=training_output/

RUN_TAG="${MODEL_TAG}_bs${BATCH_SIZE}_ep${EPOCHS}_lr$(printf %.0e $LEARNING_RATE)"
LOGS_DIR="logs"
mkdir -p $LOGS_DIR
LOG_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.out"
ERR_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.err"
exec > $LOG_FILE 2> $ERR_FILE

module purge
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh
conda init
conda activate tensorflow-gpu-v2.8

START_TIME=$(date +%s)

python train.py --root_dir $ROOT_DIR --global_max $GLOBAL_MAX --batch_size $BATCH_SIZE --epochs $EPOCHS --learning_rate $LEARNING_RATE
