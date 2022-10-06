export CUDA_VISIBLE_DEVICES=${1:-0}
export EXPERIMENT_PATH=/home/nas4_user/sungwonhwang/logs/hypernerf
EXPERIMENTS=(\
"edit_mkimv1_11-09-2022_20:50:15" \
"edit_mkimv1_12-09-2022_19:48:49" \
"edit_mkimv1_17-09-2022_14:03:03" \
"edit_mkimv1_14-09-2022_17:45:43" \
"edit_mkimv1_04-10-2022_15:36:04" \
"edit_mkimv1_17-09-2022_14:02:12" \
"edit_mkimv1_14-09-2022_17:44:18" \
"edit_mkimv1_15-09-2022_16:43:46" \
)

DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mkimv1

for i in "${!EXPERIMENTS[@]}"; do
    echo $EXPERIMENT_PATH/${EXPERIMENTS[i]}
    python eval_edit.py \
        --base_folder $EXPERIMENT_PATH/${EXPERIMENTS[i]} \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_lip_mkim.gin
done