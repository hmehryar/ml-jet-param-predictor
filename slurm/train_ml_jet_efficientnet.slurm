#!/bin/bash
#SBATCH --job-name=ml-jet-train-efficientnet
#SBATCH --partition=ecscp
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=504:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=express

# MODEL_TAG="EfficientNet"
# BATCH_SIZE=512
# EPOCHS=50
# LEARNING_RATE=0.001
# GLOBAL_MAX=121.79151153564453
# ROOT_DIR=~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000000_balanced_unshuffled
# OUTPUT_DIR=training_output/

CONFIG_PATH="$1"
# CONFIG_PATH="config/efficientnet_bs512_ep50_lr1e-04_ds1000.yml"
# CONFIG_PATH="config/efficientnet_bs512_ep50_lr1e-04_ds1000.yml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "❌ Config file not found: $CONFIG_PATH"
  exit 1
fi

# === Extract YAML Fields Using Python ===
get_yaml_value() {
  python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['$1'])"
}
MODEL_TAG=$(get_yaml_value model_tag)
BATCH_SIZE=$(get_yaml_value batch_size)
EPOCHS=$(get_yaml_value epochs)
LEARNING_RATE=$(get_yaml_value learning_rate)

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
# conda activate tensorflow-gpu-v2.8
conda activate pytorch

START_TIME=$(date +%s)
jupyter nbconvert --to script train_efficientnet.ipynb --output train_efficientnet_${SLURM_JOB_ID}
# python train_efficientnet.py --root_dir $ROOT_DIR --global_max $GLOBAL_MAX --batch_size $BATCH_SIZE --epochs $EPOCHS --learning_rate $LEARNING_RATE --model_tag $MODEL_TAG --output_dir $OUTPUT_DIR
python train_efficientnet_${SLURM_JOB_ID}.py --config "$CONFIG_PATH"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "[INFO] Training completed in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
echo "[INFO] SLURM Job finished on: $(date)"
