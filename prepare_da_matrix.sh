# for dataset in ml-100k ml-1m pinterest lastfm amazon-magazine-subscriptions-18 anime
# do
# for tau_da in 0.3 1.0
# do
# python3 prepare_da_matrix.py --dataset $dataset --tau_da $tau_da --seed 2040
# done
# done
# python3 prepare_da_matrix.py --dataset=gowalla-merged --seed=2040 --tau_da=1.0 --user_inter_num_interval='[10, inf)' --item_inter_num_interval='[10, inf)'
# python3 prepare_da_matrix.py --dataset=gowalla-merged --seed=2040 --tau_da=0.3 --user_inter_num_interval='[10, inf)' --item_inter_num_interval='[10, inf)'
# mainly run: ml-100k ml-1m pinterest gowalla-merged amazon-beauty | lastfm amazon-magazine-subscriptions-18

for tau_da in 0.3 1.0
do
python3 prepare_da_matrix.py --dataset=amazon-beauty --seed=2040 --tau_da=$tau_da --user_inter_num_interval='[5, inf)' --item_inter_num_interval='[5, inf)'
done

for seed in 2040 2033 2050 2060 2070
do
for tau_da in 0.3 1.0
do
for dataset in ml-100k ml-1m pinterest amazon-magazine-subscriptions-18 lastfm # anime ta-feng-merged
do
python3 prepare_da_matrix.py --dataset=$dataset --seed=$seed --tau_da=$tau_da
done
python3 prepare_da_matrix.py --dataset=amazon-beauty --seed=$seed --tau_da=$tau_da --user_inter_num_interval='[5, inf)' --item_inter_num_interval='[5, inf)'
python3 prepare_da_matrix.py --dataset=gowalla-merged --seed=$seed --tau_da=$tau_da --user_inter_num_interval='[10, inf)' --item_inter_num_interval='[10, inf)'
done
done
