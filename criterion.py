from networkit import *
import networkx as nx
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import pickle
import scipy.sparse as sp
import copy
import random
import numpy as np
import torch


def ranking_correlation(y_out,true_val,node_num,model_size, k=10):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    k = int(node_num*(k/100))
    top_k_predict = predict_arr.argsort()[::-1][:k]
    top_k_true = true_arr.argsort()[::-1][:k]
    
    acc = 0
    for item in top_k_predict:
      if item in top_k_true:
        acc += 1
    perc_acc = (acc/k)*100

    return kt, perc_acc


def loss_cal(y_out,true_val,num_nodes,device,model_size):

    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    
    _,order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes*20

    ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    

    rank_measure=torch.sign(-1*(ind_1-ind_2)).float()
        
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
        

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
 
    return loss_rank

