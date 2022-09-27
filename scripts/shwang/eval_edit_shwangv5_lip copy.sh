export CUDA_VISIBLE_DEVICES=${1:-0}
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/${2:-'edit_shwangv5_10-08-2022_16:39:55'}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5
python eval_edit.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/edit_test_local_lip.gin