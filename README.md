This was done as part of a project for course CS328: Introduction to Data Science, at IIT Gandhinagar.

## Fast and Approximate Network analysis using Graph Neural Networks (GNNs)

In network analysis, the problem of finding influential nodes has high theoretical and practical significance. Centrality measures are common schemes that aims to find these influential nodes in a network. However, deterministically computing these values for individual node takes a lot of time and are computationally expensive. This problem elevates when the analysis is done for larger graphs. Getting an approximate estimates of these centrality measures with much lower time would be of great significance in network analysis. In this project, we aim to do the same using Graph Neural Networks (GNNs). We train a GNN based model on a synthetically generated dataset consisting of variety of networks to approximate these centrality values. Our tests on complex real networks datasets shows that GNN gives promising results in predicting these values along with a huge speed-up of nearly 80-times when compared to the currently used deterministic algorithms.

Please refer to [Report](./report.pdf) for more details on the project. 



### Dependencies

* PyTorch (Above 1.4 recommended)

* NetworkX (Loading Graphs)

* Networkit (Sparse Matrix Conversion)

* Other: Scipy, numpy & pickle

  

### Data

You can get the data as follows: 

* Download Graphs from [here](https://drive.google.com/drive/folders/1EAb4GFIUoFJi50vtWfuJl0yhx2vs1Ijc?usp=sharing).
* Download different types of networks from [here](https://drive.google.com/drive/folders/1oWFLLRMcucU5Copc7LhWBN_MBHvMx7Kt?usp=sharing) 

Download the above files and move it to the `./data` directory.



### Quick Look

`./data`: Data Directory

`betweenness.py`:   Train GNN on betweenness centrality.

`degree.py`:               Train GNN on Degree centrality.

`Closeness.py`:        Train GNN on closeness centrality.

`GNN_layers.py`:      Embedding Layer and GNN layer defined.

`model.py` :                 GNN model defined

`criterion.py`:         Margin-Rank Loss defined

`utils.py` :                  Helper function including Adjacency, Sparse, Torch Tensor conversion 



### References

```reStructuredText
Sunil Kumar Maurya, Xin Liu, and Tsuyoshi Murata. 
Fast approximations of betweenness centrality with graph neural networks. 
In Proceedings of the 28th ACM International Conference on Information and Knowledge Management,
CIKM ’19, page. doi: 10.1145/3357384.3358080. URL https://doi.org/10.1145/3357384.3358080.
```

