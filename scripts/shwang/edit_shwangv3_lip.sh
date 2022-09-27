export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwangv3
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/edit_shwangv3
python edit.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/edit_test_local_lip.gin