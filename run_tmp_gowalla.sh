dataset=$1
seed=$2
gpu_id=$3
tau_da=1.0
beta_discount_factor=10
reg_weight=1e-3
learning_rate=1e-3
for model in DirectAU NCL DGCF SGL LightGCN BPR NGCF
do
for da in full ui anchor # none
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf_gowalla.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id
done
done
