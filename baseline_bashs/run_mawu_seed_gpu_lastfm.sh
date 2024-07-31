da=none
model=MAWU
learning_rate=0.001
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=0.01 # just for occupy
# 3/26 crashed at --gpu_id=2 --gamma1=0.1 --gamma2=5 --weight_decay=1e-4 --encoder=LightGCN --n_layers=2
# 3/30 accientlly kill at encoder:"LightGCN" gamma1:5 gamma2:2 gpu_id:1 learning_rate:0.001 model:"MAWU" n_layers:2 reg_weight:0.01 weight_decay:0.000001
for dataset in lastfm
do
for n_layers in 2
do
for encoder in LightGCN 
do
for gamma1 in 5 # 0.5 1 2 5 0.1
do
for gamma2 in 2 5 # 0.1 0.5 1 2 5
do 
for weight_decay in 1e-6 1e-8 1e-4 0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--gamma1=$gamma1 --gamma2=$gamma2 --weight_decay=$weight_decay --encoder=$encoder --n_layers=$n_layers # LGCN-2 is recommended by the original paper
done
done
done
done
done
done

for dataset in lastfm
do
for encoder in MF 
do
for gamma1 in 0.5 1 2 5 0.1
do
for gamma2 in 0.1 0.5 1 2 5
do 
for weight_decay in 1e-6 1e-8 1e-4 0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--gamma1=$gamma1 --gamma2=$gamma2 --weight_decay=$weight_decay --encoder=$encoder --n_layers=$n_layers # LGCN-2 is recommended by the original paper
done
done
done
done
done

for dataset in lastfm
do
for n_layers in 3
do
for encoder in LightGCN MF 
do
for gamma1 in 0.5 1 2 5 0.1
do
for gamma2 in 0.1 0.5 1 2 5
do 
for weight_decay in 1e-6 1e-8 1e-4 0
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--gamma1=$gamma1 --gamma2=$gamma2 --weight_decay=$weight_decay --encoder=$encoder --n_layers=$n_layers # LGCN-2 is recommended by the original paper
done
done
done
done
done
done
