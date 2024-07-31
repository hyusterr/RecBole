dataset=amazon-magazine-subscriptions-18
seed=$1
gpu_id=$2
#tau_da=1.0
#beta_discount_factor=10
#reg_weight=1e-3
#learning_rate=1e-3
# 3/18 killed at --DA_sampling=full --config_files=./configs/general_dagcf.yaml --seed=2050 --n_layers=3 --reg_weight=1e-3 --learning_rate=1e-3 --gpu_id=2 --tau_da=0.3 --beta_discount_factor=200
for reg_weight in 1e-4 # 1e-5 1e-3 # recommended by original paper
do
for learning_rate in 1e-3 # 1e-4 1e-5 # recommended by original paper
do
for n_layers in 4
do
for model in HMLET
do
# original models
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=none --config_files=./configs/general_dagcf.yaml --seed=$seed --n_layers=$n_layers --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id --tau_da=1.0 --beta_discount_factor=0 # for occupy
for tau_da in 0.3 1.0
do
for beta_discount_factor in 200 2 10 50 100 
do
for da in ui anchor full
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --n_layers=$n_layers --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor
done
done
done
done
done
done
done
