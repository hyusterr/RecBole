da=none
model=MAWU
learning_rate=0.001
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=0.01 # just for occupy
# 3/26 crashed at --reg_weight=0.01 --learning_rate=0.001 --gpu_id=2 --gamma1=0.1 --gamma2=5 --weight_decay=1e-4 --encoder=LightGCN --n_layers=2
for dataset in amazon-magazine-subscriptions-18
do
for n_layers in 2 3 
do
for encoder in LightGCN MF 
do
for gamma1 in 0.5 1 2 5 0.1
do
for gamma2 in 0.1 0.5 1 2 5
do 
for weight_decay in 0 1e-4 1e-6 1e-8 
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--gamma1=$gamma1 --gamma2=$gamma2 --weight_decay=$weight_decay --encoder=$encoder --n_layers=$n_layers # LGCN-2 is recommended by the original paper
done
done
done
done
done
done
