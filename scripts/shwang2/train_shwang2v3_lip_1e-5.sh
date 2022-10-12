export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwang2v3
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/shwang2v3_1e-5
python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_lip_shwang2_1e-5.gin
