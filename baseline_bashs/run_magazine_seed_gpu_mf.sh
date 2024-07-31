dataset=amazon-magazine-subscriptions-18
seed=$1
gpu_id=$2
# BPR
for reg_weight in 1e-3 1e-4 1e-5
do
for learning_rate in 1e-3 1e-4 1e-5
do
# BPR do not need layers
python3 run_dagcf.py --dataset=$dataset --model=BPR --DA_sampling=none --config_files=./configs/general_dagcf.yaml --seed=$seed --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--tau_da=1.0 --beta_discount_factor=0 # for occupy
for tau_da in 1.0 0.3
do
for beta_discount_factor in 2 10 50 100 200
do
for da in full ui anchor
do
python3 run_dagcf.py --dataset=$dataset --model=BPR --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id
done
done
done
done
done
