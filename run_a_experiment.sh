model=$1
dataset=$2
seed=$3
for DA_sampling in none full ui anchor
do
for tau_da in 1.0 0.3
do
python3 run_hyper_dagcf.py --model=$model --dataset=$dataset --DA_sampling=none --config_files=./configs/general_dagcf.yaml --params_file=./hyper/$model.dagcf.hyper --tau_da=1.0 --gpu_id=1 --seed=$seed --num_concurrent=0.5
done
done
