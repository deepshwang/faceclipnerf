export CUDA_VISIBLE_DEVICES=${1:-0}
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf
EXPERIMENTS=(\
"edit_jhyungv1_15-09-2022_11:13:56" \
"edit_jhyungv1_16-09-2022_10:16:59" \
"edit_jhyungv1_20-09-2022_10:37:46" \
"edit_jhyungv1_18-09-2022_08:26:35" \
"edit_jhyungv1_19-09-2022_07:27:50" \
"edit_jhyungv1_15-09-2022_11:14:13" \
"edit_jhyungv1_17-09-2022_09:10:56" \
"edit_jhyungv1_20-09-2022_10:37:48" \
"edit_jhyungv1_18-09-2022_08:09:39" \
"edit_jhyungv1_19-09-2022_07:08:29" \
)

DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1

for i in "${!EXPERIMENTS[@]}"; do
    echo $EXPERIMENT_PATH/${EXPERIMENTS[i]}
    python eval_edit.py \
        --base_folder $EXPERIMENT_PATH/${EXPERIMENTS[i]} \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_lip_jhyung.gin
done