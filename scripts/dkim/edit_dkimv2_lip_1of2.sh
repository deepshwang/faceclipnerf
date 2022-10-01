#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
PROJ_NAME=dkimv2
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/$PROJ_NAME

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/$PROJ_NAME

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_$PROJ_NAME

EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_$(date +"%d-%m-%Y_%T").txt

#TEXT_PROMPTS=("happy face" "sad face" "surprised face" "scared face" "disgusted face")
TEXT_PROMPTS=("surprised face" "angry face" "disappointed face" "sad face")
ALPHATV_LAMBDA=("100" "100" "1" "100")
REFERENCE_WARP_ID=5
ANCHOR_EMBEDDING_IDS=5,128,162,274
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}" | AlphaTV: "${ALPHATV_LAMBDA[i]}" | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_lip_dkim.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --lambda_alphatv "${ALPHATV_LAMBDA[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID \
        --anchor_embedding_ids $ANCHOR_EMBEDDING_IDS
done
