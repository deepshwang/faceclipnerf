#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mkimv1

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/mkimv1_nerfies

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_mkimv1_nerfies

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_nerfies_$(date +"%d-%m-%Y_%T").txt

TEXT_PROMPTS=("sleeping face") 
REFERENCE_WARP_ID=9
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit_nerfies.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_nerfies_test_local_lip_mkim.gin \
        --target_text_prompt "${TEXT_PROMPTS[i]}"
done
