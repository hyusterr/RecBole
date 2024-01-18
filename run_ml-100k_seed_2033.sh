dataset=ml-100k
seed=2033
tau_da=1.0
beta_discount_factor=10
reg_weight=1e-3
learning_rate=1e-3
for model in DirectAU # LightGCN BPR NCL NGCF DGCF SGL
do
for da in full ui anchor # none
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/directau.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=0
done
done
