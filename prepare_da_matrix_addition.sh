for seed in 2033 2040 2050 2060 2070
do
for tau_da in 1.0 0.3
do
python3 prepare_da_matrix.py --dataset=lastfm --seed=$seed --tau_da=$tau_da
done
done

for seed in 2033 2040 2050 2060 2070
do
for tau_da in 1.0 0.3
do
python3 prepare_da_matrix.py --dataset=gowalla-merged --seed=$seed --tau_da=$tau_da --user_inter_num_interval='[10, inf)' --item_inter_num_interval='[10, inf)'
python3 prepare_da_matrix.py --dataset=ta-feng-merged --seed=$seed --tau_da=$tau_da
done
done
