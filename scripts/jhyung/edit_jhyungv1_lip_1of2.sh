#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
PROJ_NAME=jhyungv1
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/$PROJ_NAME

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/$PROJ_NAME

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_$PROJ_NAME

EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_$(date +"%d-%m-%Y_%T").txt

TEXT_PROMPTS=("disappointed face")
ALPHATV_LAMBDA=("1")
REFERENCE_WARP_ID=3
ANCHOR_EMBEDDING_IDS=3,109,199,289
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_lip_jhyung.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --lambda_alphatv "${ALPHATV_LAMBDA[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID \
        --anchor_embedding_ids $ANCHOR_EMBEDDING_IDS
done
