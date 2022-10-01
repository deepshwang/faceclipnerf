#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mkimv2

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/mkimv2_nolip

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_mkimv2

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_vanilla_$(date +"%d-%m-%Y_%T").txt

TEXT_PROMPTS=("angry face") 
REFERENCE_WARP_ID=79
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit_vanilla3.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_nolip_mkim_vanilla3.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID
done