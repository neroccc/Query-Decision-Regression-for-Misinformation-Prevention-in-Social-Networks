# Query-Decision-Regression-for-Misinformation-Prevention-in-Social-Networks
This repository contains the code for the experiment in the paper entitled Query-Decision-Regression-for-Misinformation-Prevention-in-Social-Networks, accepted in proceedings of the 13th International Conference on Computational Data and Social Networks, CSoNet 2024.

This repository consists of 2 methods, namely, MetaLearner, StratLearner [(StratLearner)]([https://github.com/Microsoft/Graphormer](https://github.com/cdslabamotong/stratLearner/tree/master)). The implement of MetaLeraner, StratLearner are in the MetaLearner folder in this repository

## MetaLearner
We have the code for training MetaLeraner as Soup_train.py. It includes the implementation of the Sampling modification and Function estimation. We also have the code reproducing the results shown in the paper as auto.py 
### Usage
#### Run 
```
cd MetaLearner
python3 Soup_train.py
python3 auto.py
```

#### Data
The data used in the paper is in the folder named `data/`.
Each folder corresponds to a social network structure. The query-decision pairs and ground truth social network can be found in it.
## StratLearner
We have the code for training MetaLeraner as train.py. We also have the code reproducing the results shown in the paper in sauto.py  
### Usage
#### Run 
```
cd MetaLearner
python3 train.py
python3 sauto.py
```
