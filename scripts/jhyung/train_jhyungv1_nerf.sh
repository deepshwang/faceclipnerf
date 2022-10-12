export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/jhyungv1_nerf
python nerf_train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_lip_jhyung_nerf.gin