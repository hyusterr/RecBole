# da=full
seed=2033
reg_weight=1e-3
learning_rate=1e-3
gpu_id=0
for dataset in ml-100k lastfm ml-1m amazon-magazine-subscriptions-18 pinterest
do
for model in NCL DirectAU BPR LightGCN SGL DGCF
do
for da in anchor
do
for tau_da in 1.0 0.3
do
for beta_discount_factor in 2 10 50 100 200
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$da --config_files=./configs/general_dagcf.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id --num_clusters=100 
done
done
done
done
done
