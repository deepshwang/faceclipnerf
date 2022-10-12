export CUDA_VISIBLE_DEVICES=${1:-0}
LR=${2:-""}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/shwang2v2
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/shwang2v2${LR}
python eval.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_lip_shwang2.gin \
    --reference_warp_id 5