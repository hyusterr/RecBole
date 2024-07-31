da=none
model=XSimGCL
#learning_rate=0.001 # use default of recbole
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=1e-4 # default of xsimgcl
for dataset in ml-1m lastfm ml-100k pinterest amazon-magazine-subscriptions-18
do
for learning_rate in 1e-3
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id
done
done
