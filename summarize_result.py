import pandas as pd 
import wandb
api = wandb.Api()

seed = 2040
RUNS_DICT = {
    "ml-100k": f"yushuang/sigir_dagcf_{seed}_ml-100k",
    "ml-1m": f"yushuang/sigir_dagcf_{seed}_ml-1m",
    "lastfm": f"yushuang/sigir_dagcf_{seed}_lastfm",
    "magazine": f"yushuang/sigir_dagcf_{seed}_amazon-magazine-subscriptions-18",
    "pinterest": f"yushuang/sigir_dagcf_{seed}_pinterest",
}
MODEL_ORDER = [
    'BPR_none', 'BPR_full', 'BPR_ui', 'BPR_anchor',
    'LightGCN_none', 'LightGCN_full', 'LightGCN_ui', 'LightGCN_anchor',
    'LRGCCF_none', 'LRGCCF_full', 'LRGCCF_ui', 'LRGCCF_anchor',
    'DirectAU_none', 'MAWU_none', 
    'SimpleX_none', 'DiffRec_none', 
    'NCL_none', 'SGL_none',
]
METIRC_ORDER = [
    'eval/recall@20', 'eval/recall@50', 'eval/recall@100',
    'eval/ndcg@20', 'eval/ndcg@50', 'eval/ndcg@100',
]

def get_runs_performance(runs_name, dataset_name=None):
    runs = api.runs(runs_name)
    runs_list = []
    print(type(runs))
    print(len(runs))
    dataset = dataset_name if dataset_name else runs_name.split('_')[-1]
    for run in runs: 
        tmp = []
        if run.state != 'finished':
            print("Not finished:", run.name)
            continue
        name_clean = "_".join(run.name.split('_')[:2])
        tmp.append(name_clean)
        try:
            tmp.append(run.summary['valid/ndcg@50'])
        except KeyError:
            print("No valid/ndcg@50", run.name)
            try:
                tmp.append(run.summary['eval/ndcg@50'])
            except KeyError:
                print("No eval/ndcg@50", run.name)
                continue


        tmp.append(run.config['model'])
        tmp.append(run.config['final_config_dict']['tau_da'])
        tmp.append(run.config['final_config_dict']['DA_sampling'])
        tmp.append(run.config['final_config_dict']['beta_discount_factor'])
        try:
            tmp.append(run.config['final_config_dict']['n_layers'])
        except KeyError:
            tmp.append(None)
        tmp.append(dataset)
        tmp.append(run.summary['eval/recall@20'])
        tmp.append(run.summary['eval/recall@50'])
        tmp.append(run.summary['eval/recall@100'])
        tmp.append(run.summary['eval/ndcg@20'])
        tmp.append(run.summary['eval/ndcg@50'])
        tmp.append(run.summary['eval/ndcg@100'])
        try:
            tmp.append(run.summary['train_step'])
        except KeyError:
            tmp.append(None)
            print("No train_step", run.name)
        try:
            tmp.append(run.summary['_wandb']['runtime'])
        except KeyError:
            tmp.append(None)
            print("No runtime", run.name)

        runs_list.append(tmp)
    
    runs_df = pd.DataFrame(runs_list, columns=['name', 'valid/ndcg@50', 'model', 'tau_da', 'DA_sampling', 'beta_discount_factor', 'n_layers', 'dataset', 'eval/recall@20', 'eval/recall@50', 'eval/recall@100', 'eval/ndcg@20', 'eval/ndcg@50', 'eval/ndcg@100', 'train_step', 'runtime'])

    # group the df by name, sort by valid/ndcg@50, and get the best model
    runs_df = runs_df.groupby('name').apply(lambda x: x.sort_values('valid/ndcg@50', ascending=False).head(1)).reset_index(drop=True)
        
    runs_df.index = runs_df['name']
    
    model_order = []; model_we_have = runs_df['name'].values
    for m in MODEL_ORDER:
        if m in model_we_have:
            model_order.append(m)

    

    pretty_df = runs_df[METIRC_ORDER].T[model_order].copy()
    pretty_df['metric'] = pretty_df.index.map(lambda x: x.split('/')[1].split('@')[0])
    pretty_df['K'] = pretty_df.index.map(lambda x: x.split('/')[1].split('@')[1])
    pretty_df = pretty_df.reset_index(drop=True)
    pretty_df['dataset'] = dataset
    return pretty_df[['dataset', 'metric', 'K'] + model_order]

def aggregate_result_from_different_datasets():
    result_df = pd.DataFrame()
    for dataset, runs_name in RUNS_DICT.items():
        runs_df = get_runs_performance(runs_name, dataset)
        result_df = pd.concat([result_df, runs_df], axis=0)
    return result_df


# Project is specified by <entity/project-name>
runs = api.runs("yushuang/sigir_dagcf_2040_amazon-magazine-subscriptions-18")
runs_df = get_runs_performance("yushuang/sigir_dagcf_2040_amazon-magazine-subscriptions-18")
runs_df.to_csv("project.csv")

result_df = aggregate_result_from_different_datasets()
result_df.to_csv(f"all-exp-seed-{seed}.csv")
