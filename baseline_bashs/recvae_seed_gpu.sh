da=none
model=RecVAE
#learning_rate=0.001 # use default of recbole
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=1e-5 # default of simplex
for dataset in lastfm amazon-magazine-subscriptions-18 pinterest ml-1m ml-100k
do
for learning_rate in 0.01 0.005 0.001 0.0005 0.0001
do
for latent_dimension in 64 100 128 150 200 256 300 400 512
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--latent_dimension=$latent_dimension
done
done
done
