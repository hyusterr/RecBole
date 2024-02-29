import json
import numpy as np
import matplotlib.pyplot as plt

from recbole.evaluator.base_metric import TopkMetric
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk, full_sort_scores


model_file_name = 'case/ml-1m-LightGCN-none-Feb-29-2024_18-38-34.pth'
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=model_file_name,
)

print(config)

uid_series = dataset.token2id(dataset.uid_field, ['196', '186'])
score = full_sort_scores(uid_series, model, test_data, device=config['device'])
print(score)  # score of all items
print(score[0, dataset.token2id(dataset.iid_field, ['242', '302'])])
# score of item ['242', '302'] for user '196'.

topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
print(topk_score)  # scores of top 10 items
print(topk_iid_list)  # internal id of top 10 items
external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
print(external_item_list)  # external tokens of top 10 items
