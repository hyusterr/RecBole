da=none
model=SimSGL
learning_rate=0.001 # use default of recbole
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=1e-5 # default of simplex
for dataset in ml-100k ml-1m lastfm amazon-magazine-subscriptions-18 pinterest
do
for ssl_tau in 0.1 0.2 0.5 1.0
do
for drop_ratio in 0 0.1 0.2 0.4 0.5
do 
for ssl_weight in 0.005 0.05 0.1 0.5 1.0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--ssl_tau=$ssl_tau --drop_ratio=$drop_ratio --ssl_weight=$ssl_weight
done
done
done
done
