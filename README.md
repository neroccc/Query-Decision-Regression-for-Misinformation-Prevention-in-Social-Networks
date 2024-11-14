# Query-Decision-Regression-for-Misinformation-Prevention-in-Social-Networks
This repository contains the code for the experiment in the paper entitled Query-Decision-Regression-for-Misinformation-Prevention-in-Social-Networks, accepted in proceedings of the 13th International Conference on Computational Data and Social Networks, CSoNet 2024.

This repository consists of 2 methods, namely, MetaLearner, StratLearner [(StratLearner)]([https://github.com/Microsoft/Graphormer](https://github.com/cdslabamotong/stratLearner/tree/master)). The implement of MetaLeraner, StratLearner are in the MetaLearner folder in this repository

## MetaLearner
We have the code for training MetaLeraner is Soup_train.py. It includes the implement of the Sampling modification and Function estimation. We also have the code reproduce the results shown in the paper is auto.py 
### Usage
#### Run 
```
cd MetaLearner
python3 Soup_train.py
python3 auto.py
```

#### Data
The data used in the paper isin folder named `data/`.
Each folder is corresponded to a social network structure. The query-decison pairs and ground truth social network can be found in it.
## StratLearner
We have the code for training MetaLeraner is train.py. We also have the code reproduce the results shown in the paper is sauto.py  
### Usage
#### Run 
```
cd MetaLearner
python3 train.py
python3 sauto.py
```
