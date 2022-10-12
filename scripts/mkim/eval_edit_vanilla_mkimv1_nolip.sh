export CUDA_VISIBLE_DEVICES=${1:-0}
export DATASET_PATH=/home/nas4_user/sungwonhwang/data/hypernerf/mkimv1
EXPERIMENT_ROOT="/home/nas2_userG/junhahyung/faceclipnerf_logs/out"
EXPERIMENTS=("mkimv1_08-10-2022_17:11:06" \
"mkimv1_08-10-2022_17:11:26" \
"mkimv1_08-10-2022_17:11:30" \
"mkimv1_08-10-2022_17:11:41" \
"mkimv1_08-10-2022_17:11:51" \
"mkimv1_08-10-2022_17:11:56" \
"mkimv1_08-10-2022_17:12:01" \
"mkimv1_08-10-2022_17:12:07" \
)

for i in "${!EXPERIMENTS[@]}"; do
    echo "${EXPERIMENTS[i]}"
    python eval_edit_vanilla.py \
        --base_folder $EXPERIMENT_ROOT/"${EXPERIMENTS[i]}" \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/edit_test_local_nolip_mkim_vanilla2.gin
done