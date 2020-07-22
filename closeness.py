 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
from criterion import *
import random
import torch.nn as nn
from model import GNN2
torch.manual_seed(20)


gtype = "SF"

a = input("Please Enter a graph type. 'SF' for Scale-Free, 'ER' for Erdos-Renyi, 'GRP' for Gaussian:  ")
gtype=a

if gtype == "SF":
    data_path = "./data/types/SF/closeness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./data/types/ER/closeness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./data/data_splits/GRP/closeness/"
    print("Gaussian Random Partition graphs selected.")



print(f"Process [1]: Loading Graphs.")

with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,cc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,cc_mat_test = pickle.load(fopen)

model_size = 10000
print(f"Sparse Matrix conversion.")

list_adj_train,list_adj_mod_train = graph_to_adj_close(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_mod_test = graph_to_adj_close(list_graph_test,list_n_seq_test,list_num_node_test,model_size)



def train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_mod = list_adj_mod_train[i]
        adj = adj.to(device)
        adj_mod = adj_mod.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_mod)
        true_arr = torch.from_numpy(cc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_mod_test,list_num_node_test,bc_mat_test, k=10):
    model.eval()
    loss_val = 0
    list_kt = list()
    list_acc = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_mod = list_adj_mod_test[j]
        adj=adj.to(device)
        adj_mod = adj_mod.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_mod)
    
        
        true_arr = torch.from_numpy(cc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt, perc_acc = ranking_correlation(y_out,true_val,num_nodes,model_size, k)
        list_kt.append(kt)
        list_acc.append(acc)

    print(f"Top {k}-Hit Accuracy: {np.mean(np.array(list_acc))} and std: {np.std(np.array(list_acc))}")
    print(f"Kendall-Tau scores is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")




hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")
model = GNN_Close(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 15

print("Training")
print(f"Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train)

    with torch.no_grad():
        test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test, k=10)
