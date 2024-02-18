da=none
model=DirectAU
learning_rate=0.001
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=0.01 # just for occupy
for dataset in ml-100k ml-1m lastfm amazon-magazine-subscriptions-18 pinterest
do
for encoder in LightGCN MF 
do
for gamma in 1 2 5 10 0.01 0.1 0.2 0.5
do 
for weight_decay in 0 1e-4 1e-6 1e-8 
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--gamma=$gamma --weight_decay=$weight_decay --encoder=$encoder --n_layers=2 # LGCN-2 is recommended by the original paper
done
done
done
done
