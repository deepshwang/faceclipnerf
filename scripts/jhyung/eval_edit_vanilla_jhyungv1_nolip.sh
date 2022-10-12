export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1
EXPERIMENT_ROOT="/home/nas2_userG/junhahyung/faceclipnerf_logs/out"
EXPERIMENTS=("jhyungv1_06-10-2022_18:51:28" \
             "jhyungv1_06-10-2022_18:54:17" \
             "jhyungv1_06-10-2022_18:54:22" \
             "jhyungv1_06-10-2022_18:54:27" \
             "jhyungv1_06-10-2022_18:54:33" \
             "jhyungv1_06-10-2022_18:54:39" \
             "jhyungv1_06-10-2022_18:54:44" \
             "jhyungv1_06-10-2022_18:54:51" \
)

for i in "${!EXPERIMENTS[@]}"; do
    echo "${EXPERIMENTS[i]}"
    python eval_edit_vanilla.py \
        --base_folder $EXPERIMENT_ROOT/"${EXPERIMENTS[i]}" \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_nolip_jhyung_vanilla2.gin
done