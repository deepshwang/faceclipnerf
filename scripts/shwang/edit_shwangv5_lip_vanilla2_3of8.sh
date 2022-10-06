#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/shwangv5_nolip

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_shwangv5

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_vanilla2_$(date +"%d-%m-%Y_%T").txt

TEXT_PROMPTS=("surprised face") 
REFERENCE_WARP_ID=9
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit_vanilla2.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_nolip_vanilla2.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID
done
