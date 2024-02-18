da=none
model=NCL
learning_rate=0.001 # NCL's default value # not specified, use value from overall.yaml
seed=$1
gpu_id=$2
tau_da=1.0 # just for occupy
beta_discount_factor=0 # just for occupy
reg_weight=0.0001 # NCL's default value
for ssl_temp in 0.05 0.07 0.1
do
for ssl_reg in 1e-6 1e-7
do 
for proto_reg in 1e-6 1e-7 1e-8
do
for num_clusters in 100 1000
do
for dataset in ml-100k ml-1m lastfm amazon-magazine-subscriptions-18 pinterest
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id \
--ssl_temp=$ssl_temp --ssl_reg=$ssl_reg --proto_reg=$proto_reg --num_clusters=$num_clusters 
done
done
done
done
done
