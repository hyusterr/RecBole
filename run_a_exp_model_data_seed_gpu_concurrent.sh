model=$1
dataset=$2
seed=$3
gpu_id=$4
num_concurrent=$5

# run a single trial to get some hint of results
for da in full ui anchor
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=1.0 --beta_discount_factor=10 --reg_weight=1e-3 --learning_rate=1e-3 --gpu_id=$gpu_id
done
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=none --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=1.0 --beta_discount_factor=0 --reg_weight=1e-3 --learning_rate=1e-3 --gpu_id=$gpu_id

# run hyper-parameter tuning
python3 run_hyper_dagcf.py --model=$model --dataset=$dataset --DA_sampling=none --config_files=./configs/general_dagcf.yaml --params_file=./hyper/$model.hyper --tau_da=1.0 --gpu_id=$gpu_id --seed=$seed --num_concurrent=$num_concurrent
for DA_sampling in full ui anchor
do
for tau_da in 1.0 0.3
do
python3 run_hyper_dagcf.py --model=$model --dataset=$dataset --DA_sampling=$DA_sampling --config_files=./configs/general_dagcf.yaml --params_file=./hyper/$model.dagcf.hyper --tau_da=$tau_da --gpu_id=$gpu_id --seed=$seed --num_concurrent=$num_concurrent
done
done
