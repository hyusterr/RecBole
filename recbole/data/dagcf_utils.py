import os
import csv
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=16):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [
        pool.apply_async(
            single_source_shortest_path_length_range,
            args=(
                graph,
                nodes[
                    int(len(nodes) / num_workers * i) : int(
                        len(nodes) / num_workers * (i + 1)
                    )
                ],
                cutoff,
            ),
        )
        for i in range(num_workers)
    ]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def precompute_dist_data(train_data, config, approximate=0):
    """
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    it is the NSPD in the paper
    train_data: recbole.data.AbstractDataLoader
    :return:
    """
    # tau = config["tau_da"]
    num_nodes = train_data._dataset.item_num + train_data._dataset.user_num
    # will the train_data num info same to the original data??
    train_df = train_data._dataset.inter_feat.interaction
    print(num_nodes)
    print(train_df)
    stophere


    print(edge_index)
    graph = nx.Graph()
    edge_list = edge_index.tolist()  # np.array shape=(n_iteraction, 2)
    graph.add_edges_from(edge_list)

    n = num_nodes  # calculate on all nodes
    dists_array = np.zeros((n, n))
    dists_dict = all_pairs_shortest_path_length_parallel(
        graph, cutoff=approximate if approximate > 0 else None
    )

    max_path_length = 0
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist >= max_path_length:
                max_path_length = dist
    print("max path length", print(max_path_length))

    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                if tau == 0:  # -1 ~ 1
                    dists_array[node_i, node_j] = -(((dist / max_path_length) * 2) - 1)
                elif tau > 1:  # 0 ~ 1
                    dists_array[node_i, node_j] = 1 / (dist + tau)
    return dists_array

if __name__ == '__main__':
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    config = Config(model='BPR', dataset='ml-100k')
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    precompute_dist_data(train_data, config, approximate=0)
