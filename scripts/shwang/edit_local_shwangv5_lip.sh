#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/shwangv5

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_shwangv5

EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_local_log_$(date +"%d-%m-%Y_%T").txt

TEXT_PROMPTS=("face with closed eyes" "face with opened mouth")
ALPHATV_LAMBDA=("0" "0")
REFERENCE_WARP_ID=9
ANCHOR_EMBEDDING_IDS=9,120,155,285
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}" | lambda_alphatv: "${ALPHATV_LAMBDA[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_local_test_local_lip.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --lambda_alphatv "${ALPHATV_LAMBDA[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID \
        --anchor_embedding_ids $ANCHOR_EMBEDDING_IDS
done
