#!/bin/bash

# --- INPUT ARGUMENTS ---
# These match the order in your submit file: $(model) $(mode) $(model_path) $(algo) $(exp_id) $(diff_lr)
MODEL=$1          # ex: eat_lrg_aasist
MODE=$2           # train or eval
MODEL_PATH=$3     # required for eval, 'none' for train
ALGO=$4           # ex: 5 (RawBoost) or 0 (None)
EXP_ID=$5         # ex: exp_baseline
DIFF_LR_FLAG=$6   # ex: "--diff_lr" or "" (empty)

# --- ENVIRONMENT SETUP ---
source /etc/profile.d/modules.sh
module load cuda/12.1
source /fast/krevi/venvs/envsdd/bin/activate

# Updated directory
cd /home/krevi/EnvSDD_project2

# --- GPU LOGGING ---
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"

# --- DATASET PATHS ---
TRAIN_JSON="dev_track1_train.json"
DEV_JSON="dev_track1_valid.json"
TEST_JSON="dev_track1_valid.json"

# --- CONSTRUCT COMMAND ---
# We add --algo, --exp_id, and the optional $DIFF_LR_FLAG variable
CMD="python -u main.py \
    --train_meta_json $TRAIN_JSON \
    --dev_meta_json $DEV_JSON \
    --test_meta_json $TEST_JSON \
    --model $MODEL \
    --algo $ALGO \
    --exp_id $EXP_ID \
    $DIFF_LR_FLAG"

# --- TRAIN MODE CONFIG ---
if [[ "$MODE" == "train" ]]; then
    # Adjust batch size/workers for your specific GPU (A100/H100)
    CMD="$CMD --batch_size 32 --num_workers 8 --lr 0.0001"
fi

# --- EVAL MODE CONFIG ---
if [[ "$MODE" == "eval" ]]; then
    if [[ "$MODEL_PATH" == "none" ]]; then
        echo " ERROR: model_path is required for eval mode."
        deactivate
        exit 1
    fi
    # Faster inference batch size
    CMD="$CMD --eval --model_path $MODEL_PATH --batch_size 64"
fi

# --- EXECUTION ---
echo "------------------------------------------------"
echo "Mode: $MODE | Algo: $ALGO | ExpID: $EXP_ID | Diff-LR: '$DIFF_LR_FLAG'"
echo "Command: $CMD"
echo "------------------------------------------------"

eval $CMD

deactivate