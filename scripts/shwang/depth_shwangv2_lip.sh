export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwangv2
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_shwangv2
python extract_depth.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/edit_test_local_lip.gin