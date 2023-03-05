## CIMLA (Counterfactual Inference by Machine Learning and Attribution Models)
Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home)

## Description
CIMLA is a counterfactual inference method for identifying differential causal associations between two groups from observational data.

## Getting Started
CIMLA can be installed from PyPI, but we recommend to first create a conda environment:

```conda create --name CIMLA_env python=3.8.12```

Activate this environment and install CIMLA:

```pip install CIMLA```

## Run CIMLA
CIMLA can be used in the enviornment created above. A YAML config file (see below) is required to run CIMLA:

```cimla --config config_test.yaml```

## CIMLA Config File
The config file specifies user defined settings in four sections: data, ML, attribution, and post_process. Please refer to the manual below for more details.

### data:
* path_g1: *{string, required}* Directory of the observational data for group 1. Data should be stored with rows as samples and columns as variables, also header (column names) and row names are required. 

```yaml
data:
 path_g1: *{string, required}* Directory of the observational data for group 1. Data should be stored with rows as samples and columns as variables, also header (column names) and row names are required. 
 path_g2: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/pr_0.1_rID_1/s2/expression_s2.csv
 number_samples_g1: 3000
 number_samples_g2: 3000
 independent_parameters: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/pr_0.1_rID_1/pars/indep_pars_0.csv
 dependent_parameters: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/pr_0.1_rID_1/pars/dep_par_0.csv
 normalize: True
 test_size: 0.2
 split_random_state: 32
 #cache: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/out
 cache:
 sample_column:


ML:
 #Types: RF, MLP, XGB
 #Tasks: regression, (classification is not fully supported yet)
 type: MLP
 task: regression

 ## tree (RF and XGBoost) specific parameters
 ## Use "rmse" for XGB regression and "neg_mean_squared_error" for RF regression
 max_depth: [10, 30]
 scoring: rmse
 #scoring: neg_mean_squared_error

 ## RF specific parameters
 n_estimators: [50, 200]
 max_features: ['sqrt']
 min_samples_leaf: [1, 5]
 max_leaf_nodes: [50, 200]
 cv: 3

 ## XGBoost specific parameters
 min_child_weight: [5]
 subsample: [1]
 colsample: [1]
 eta: [0.1, 0.01]
 l2: [1, 5, 10]
 l1: [0.01, 0.1, 1]
 max_boost_round: 400
 early_stop_round: 30
 early_stop_tol: 0.01

 ## MLP specific parameters
 hidden_channels: [128,32]
 dropout: 0.5
 dense_layers_l2: 0.01


attribution:
 #Types: tree_shap, deep_shap
 type: deep_shap
 data_split: train
 data_group: 1
 data_size: 0.75
 global_type: rmsd


post_process:
 local_scores_path:
 global_scores_path: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/out
 ML_save_path:
 ML_performance_save_path: /home/payam/research/causal_inference_proj3/package_tests/src/github/test/out
 ML_performance_metric: mse

```





