seed=$1
gpu_id=$2
learning_rate=1e-3
reg_weight=1e-3
tau_da=0.3 # 0.3 1.0 # smaller value yields better results on the largest dataset (Pinterest)
beta_discount_factor=100 # 2 10 50 100 200
# larger beta_discount_factor for larger dataset
# 4/2: finish this script, beauty only wamu not finished; gowalla failed at LGCN-anchor, LGCN-full, DirectAU, WAMU

# DAGCFs
dataset=gowalla-merged
for model in LightGCN # BPR LightGCN LRGCCF
do
for DA_sampling in full anchor # ui
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$DA_sampling --config_files=./configs/general_dagcf_gowalla.yaml --seed=$seed --reg_weight=$reg_weight --learning_rate=$learning_rate --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --gpu_id=$gpu_id
done
done

# baselines
for model in DirectAU # BPR LightGCN LRGCCF DirectAU MAWU SGL NCL SimpleX DiffRec
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=none --config_files=./configs/general_dagcf_gowalla.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=0 --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id
done

dataset=amazon-beauty
# DAGCFs
: '
for model in BPR LightGCN LRGCCF
do
for DA_sampling in ui full anchor
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=$DA_sampling --config_files=./configs/general_dagcf_beauty.yaml --seed=$seed --reg_weight=$reg_weight --learning_rate=$learning_rate --tau_da=$tau_da --beta_discount_factor=$beta_discount_factor --gpu_id=$gpu_id
done
done
'


# baselines
# MAWU need huge memory...
: '
for model in MAWU # BPR LightGCN LRGCCF DirectAU MAWU SGL NCL SimpleX DiffRec
do
python3 run_dagcf.py --dataset=$dataset --model=$model --DA_sampling=none --config_files=./configs/general_dagcf_beauty.yaml --seed=$seed --tau_da=$tau_da --beta_discount_factor=0 --reg_weight=$reg_weight --learning_rate=$learning_rate --gpu_id=$gpu_id
done
'
