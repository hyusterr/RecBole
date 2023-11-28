# -*- coding: utf-8 -*-
# @Time   : 2023/11/28
# @Author : Huang, Yu-Shiang
# @Email  : F09946004@ntu.edu.tw


r"""
BSPM
################################################
Reference:
    Choi, Jeongwhan, et al. "Blurring-sharpening process models for collaborative filtering." in SIGIR 2023.
    reference code: https://github.com/jeongwhanchoi/BSPM/blob/main/bspm/model.py
"""

import torch
import torch.nn as nn
import time
from torch import nn
import scipy.sparse as sp
import numpy as np
from sparsesvd import sparsesvd

from torchdiffeq import odeint

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType, ModelType

class BSPM(GeneralRecommender):
    r"""
    BSPM is a blurring-sharpening process model for collaborative filtering.
    The method directly output a inferred user-item interaction matrix, so the implementation take itemknn as a template.
    Refer to the original implementation (BSPM_TORCH) for more details.
    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(BSPM, self).__init__(config, dataset)

        # load parameters info
        self.config = config
        self.idl_solver = self.config['solver_idl']
        self.blur_solver = self.config['solver_blr']
        self.sharpen_solver = self.config['solver_shr']
        # TODO: use logger?
        print('[ODE Solver Info]')
        print('Solver for IDL filter: ', self.idl_solver)
        print('Solver for blurring filter: ', self.blur_solver)
        print('Solver for sharpening filter: ', self.sharpen_solver)
        
        self.idl_beta = self.config['idl_beta']
        self.factor_dim = self.config['factor_dim'] # dimension of IDL filte
        print('[Parameter for IDL filter]')
        print(r"factor dimension: ",self.factor_dim)
        print(r"$\beta$: ",self.idl_beta)
        
        # hyperparameters for filters and solvers
        idl_T = self.config['T_idl']
        idl_K = self.config['K_idl']
        
        blur_T = self.config['T_b']
        blur_K = self.config['K_b']
        
        sharpen_T = self.config['T_s']
        sharpen_K = self.config['K_s']

        self.device = config['device']
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float().to(self.device)
        print("idl time: ",self.idl_times)
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float().to(self.device)
        print("blur time: ",self.blurring_times)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().to(self.device)
        print("sharpen time: ",self.sharpening_times)
        self.final_sharpening = self.config['final_sharpening']
        self.sharpening_off = self.config['sharpening_off']
        self.t_point_combination = self.config['t_point_combination']
        print("final_sharpening: ",self.final_sharpening)
        print("sharpening off: ",self.sharpening_off)
        print("t_point_combination: ",self.t_point_combination)
        
        # prepare user item interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype('float32')
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        self.adj_mat = self.interaction_matrix.tolil()


    def train(self):

        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        
        # if self.config['dataset'] != 'amazon-book':
        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)
        del ut
        del s

        # prepare linear filter (heat equation)
        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        # if self.config['dataset'] != 'amazon-book':
        # prepare 
        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)
        end = time.time()
        print('[TIME INFO] pre-processing time for BSPM', end-start)
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out


        
    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        # BSPM do not need training loss
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_test = user.to_sparse()
        # the bluring process steps is actually 1, so the author ignored ODE solvers in the torch blurring implementation, see: https://github.com/jeongwhanchoi/BSPM/issues/1
        with torch.no_grad():
            idl_out = torch.mm(batch_test, self.left_mat @  self.right_mat)
            blurred_out = torch.mm(batch_test, self.linear_Filter)
            del batch_test
            if self.sharpening_off == False:
                if self.final_sharpening == True: # late merge (BSPM-LM)
                    sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out+blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                else: # early merge (BSPM-EM)
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)

        if self.t_point_combination == True: # residual combination?
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
            else:
                U_2 =  blurred_out
                del blurred_out
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out

        if self.final_sharpening == True:
                if self.sharpening_off == False:
                    ret = U_2
                elif self.sharpening_off == True:
                    ret = self.idl_beta * idl_out + U_2
            else:
                ret = self.idl_beta * idl_out + U_2

        return ret
