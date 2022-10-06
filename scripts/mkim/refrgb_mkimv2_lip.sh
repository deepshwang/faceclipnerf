export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mkimv2
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/mkimv2
REF_WARP_ID=79
python extract_refrgb.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_lip_mkim.gin \
    --reference_warp_id $REF_WARP_ID