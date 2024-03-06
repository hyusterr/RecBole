import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from recbole.evaluator.base_metric import TopkMetric
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.trainer import Trainer


def case_study(model_file_name):
    # model_file_name = 'case/ml-1m-LightGCN-none-Feb-29-2024_18-38-34.pth'
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file_name,
    )

    DA_matrix = model.data_dist_matrix
    # exclude history items in training set and validation set
    hit_data = []
    neg_data = []
    for batched_data in test_data:
        interaction, history_index, positive_u, positive_i = batched_data
        history_index = history_u, history_i
        all_items = np.arange(dataset.item_num)
        uninteracted_items = np.setdiff1d(all_items, history_i)
        pos_item = positive_i
        neg_item = np.setdiff1d(uninteracted_items, pos_item)

        # get the DA matrix of positive items, index (uid, iid)
        DA_pos = DA_matrix[positive_u, pos_item]
        DA_neg = DA_matrix[positive_u, neg_item]

        for i in range(len(pos_item)):
            # uid, iid, DA_score
            hit_data.append([positive_u[i], pos_item[i], DA_pos[i]])
        for i in range(len(neg_item)):
            neg_data.append([positive_u[i], neg_item[i], DA_neg[i]])

        # the score on DA matrix is the inverse, need to inverse it back


    


    # get test user id
    uid_series = dataset.token2id(dataset.uid_field, ['196', '186'])
    # get the result of full sort predict
    score = full_sort_scores(uid_series, model, test_data, device=config['device'])
    # get the answer of test set


    # Do I need to exclude the hit items in training set and validation set then calculate the performance on test data?
    # check RecBole trainer & others implementation
    # RecBole: scores[history_index] = -np.inf
    # and send to eval_collector --> after collect all test_data then get_data_struct and evaluate the struct
    # need to figure out how did RecBole get uid2history in dataloader's collate_fn
    model.to(config["device"])
    '''
    for batched_data in test_data:
        interaction, history_index, positive_u, positive_i = batched_data
        # history_index = history_u, history_i
        scores = model.full_sort_predict(interaction.to(config["device"]))
        scores = scores.view(-1, test_data._dataset.item_num)
        scores[:, 0] = -np.inf
        if history_index is not None: # remove the history items
            scores[history_index] = -np.inf
        break
    '''


def plot_metrics(wandb_summary_file, set_='eval'):
    # with open('case/LightGCN_ml-1m_files/wandb-summary.json', 'r') as f:
    with open(wandb_summary_file, 'r') as f:
        data = json.load(f)
    recalls, ndcgs = [], []
    for key, value in data.items():
        try:
            setname, info = key.split('/')
        except ValueError:
            continue
        if setname == set_:
            metric, K = info.split('@')
            if metric == 'recall':
            recalls.append([int(K), value])
        elif metric == 'ndcg':
            ndcgs.append([int(K), value])
    recalls = np.array(recalls)
    ndcgs = np.array(ndcgs)
    plt.plot(recalls[:, 0], recalls[:, 1], label='Recall')
    plt.plot(ndcgs[:, 0], ndcgs[:, 1], label='NDCG')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('metric')
    fig_name = Path(wandb_summary_file).parent / f'{set_}.png' 
    plt.savefig(fig_name)












# print(score)  # score of all items
# print(score[0, dataset.token2id(dataset.iid_field, ['242', '302'])])
# score of item ['242', '302'] for user '196'.

# topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
# print(topk_score)  # scores of top 10 items
# print(topk_iid_list)  # internal id of top 10 items
# external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
# print(external_item_list)  # external tokens of top 10 items



# full_sort_predict

# the answer in test_data

