da=none
model=SimpleX
learning_rate=0.001 # use default of recbole
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=1e-5 # default of simplex
for dataset in ml-100k ml-1m lastfm amazon-magazine-subscriptions-18 pinterest
do
for margin in 0.5 0.9 0
do
for gamma in 0.3 0.5 0.7
do 
for negative_weight in 50 10 0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--margin=$margin --gamma=$gamma --negative_weight=$negative_weight
done
done
done
done
