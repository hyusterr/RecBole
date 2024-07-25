# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2023/10/2
# @Author : Yu-Shiang Huang
# @Email  : F09946004@ntu.edu.tw

r"""
LR-GCCF
################################################

Reference:
    Chen, Lei, et al. "Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach." in AAAI 2020.

Reference code:
    https://github.com/newlei/LRGCCF
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LRGCCF(GeneralRecommender):
    r"""LRGCCF is a GCN-based recommender model.

    remove the nonlinear acitivation and the inner product in the NGCF

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, da_data_matrix):
        super(LRGCCF, self).__init__(config, dataset, da_data_matrix)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of LRGCCF
        self.n_layers = config["n_layers"]
        # int type:the layer num of LRGCCF
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.Wlayers = torch.nn.ModuleList()
        for layer_idx in range(self.n_layers):
            self.Wlayers.append(
                torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
                # not use bias in the linear layer
            )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        (notation follows the AAAI2020 paper)
        the LRGCCF consider self connection, so the norm_adj_matrix is
        .. math::
            \tilde{A} = A + I
            \tilde{D} = \sum_{j=1}^{N} \tilde{A}_{ij} # degree matrix of \tilde{A}
            S = \tilde{D}^{-0.5} \times \tilde{A} \times \tilde{D}^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # different from LightGCN, LRGCCF consider self connection
        A_tilde = A + sp.eye(A.shape[0])

        # norm adj matrix
        sumArr = (A_tilde > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag_tilde = np.array(sumArr.flatten())[0] + 1e-7
        diag_tilde = np.power(diag_tilde, -0.5)
        D = sp.diags(diag_tilde) # actually D is D_tilde^-0.5
        S = D * A_tilde * D
        # covert norm_adj matrix to tensor
        S = sp.coo_matrix(S)
        row = S.row
        col = S.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(S.data)
        SparseS = torch.sparse.FloatTensor(i, data, torch.Size(S.shape))
        return SparseS

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for linear_layer in self.Wlayers:
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            all_embeddings = linear_layer(all_embeddings)
            embeddings_list.append(all_embeddings)
            # print("all_embeddings.shape", all_embeddings.shape)
        # resiude connection so cat the origin embedding
        lrgccf_all_embeddings = torch.cat(embeddings_list, dim=1)
        # print("lrgccf_all_embeddings.shape", lrgccf_all_embeddings.shape) 
        # (N, 256) = (N, 64) + (N, 64) + (N, 64) + (N, 64)

        user_all_embeddings, item_all_embeddings = torch.split(
            lrgccf_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        
        # add for dagcf; renew the embeddings # this is on device
        self.get_user_embedding_da = user_all_embeddings
        self.get_item_embedding_da = item_all_embeddings
        self.get_all_embedding_da = torch.cat([user_all_embeddings, item_all_embeddings])
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        return mf_loss, self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
