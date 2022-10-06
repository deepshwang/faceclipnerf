export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mhanv1
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf/mhanv1
python eval.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_lip_mhan.gin \
    --reference_warp_id 5