model=$1
dataset=$2
for seed in 2033
do
for learning_rate in 1e-3 1e-4 1e-5
do
for reg_weight in 1e-3 1e-4 1e-5
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=none --config_files=./configs/general_dagcf.yaml --seed=$seed --reg_weight=$reg_weight --learning_rate=$learning_rate --tau_da=1.0 --beta_discount_factor=2 --gpu_id=0
for da in ui full anchor
do
for beta_discount_factor in 2 10 50 100 200
do
for tau_da in 0.3 1.0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=0
done
done
done
done
done
done
