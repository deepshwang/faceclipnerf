#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
PROJ="jhyungv1"
DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/${PROJ}

REF_EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/${PROJ}

EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_${PROJ}

EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")

LOG_FILENAME=$REF_EXPERIMENT_PATH/edit_log_nerf_$(date +"%d-%m-%Y_%T").txt

#TEXT_PROMPTS=("happy face" "sad face" "surprised face" "fearful face" "disgusted face" "angry face" "face with contempt" "crying face" "happily surprised face" "sleeping face")
TEXT_PROMPTS=("scared face")
REFERENCE_WARP_ID=3
for i in "${!TEXT_PROMPTS[@]}"; do
    EXPERIMENT_PATH_WITH_TIME=${EXPERIMENT_PATH}"_"$(date +"%d-%m-%Y_%T")
    echo "Training prompt:  "${TEXT_PROMPTS[i]}"  | Result Path:  "${EXPERIMENT_PATH_WITH_TIME} >> $LOG_FILENAME
    python edit_nerf.py \
        --base_folder $EXPERIMENT_PATH_WITH_TIME \
        --ref_base_folder $REF_EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_nerf_test_local_lip.gin\
        --target_text_prompt "${TEXT_PROMPTS[i]}" \
        --reference_warp_id $REFERENCE_WARP_ID
done
