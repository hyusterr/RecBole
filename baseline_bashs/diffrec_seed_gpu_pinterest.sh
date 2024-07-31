da=none
model=DiffRec
#learning_rate=0.001 # use default of recbole
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=1e-5 # default of simplex
for dataset in pinterest
do
for learning_rate in 1e-3 1e-4 1e-5
do
for dims_dnn in '[200,600]' ['1000'] '[300]' 
# 3/26 crashed at [300]; 2; 0; 5e-3; 5e-3; 0.2 --> move [300] to the end
do
for steps in 2 5 10 50
do 
for noise_scale in 0 1e-5 1e-4 1e-3 1e-2 1e-1
do
for noise_min in 0 5e-4 1e-3 5e-3
do
for noise_max in 5e-3 1e-2
do
for w_min in 0 0.1 0.2 0.3
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--dims_dnn=$dims_dnn --steps=$steps --noise_scale=$noise_scale --noise_min=$noise_min --noise_max=$noise_max --w_min=$w_min
done
done
done
done
done
done
done
done
