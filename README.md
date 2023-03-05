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
**path_g1:** *{string, required}* 
> Directory of the observational data for group 1. Data must be stored in .csv format with rows as samples and columns as variables, also header (column names) and row names are required. 

**path_g2:** *{string, required}* 
> Directory of the observational data for group 2. Data must be stored in .csv format with rows as samples and columns as variables, also header (column names) and row names are required. 

**number_samples_g1:** *{integer, required}* 
> Number of samples (rows) in the observational data of group 1.

**number_samples_g2:** *{integer, required}* 
> Number of samples (rows) in the observational data of group 2.

**independent_parameters:** *{string, required}* 
> Directory of a text file containing the names of independent variables. Data must be stored in one column (without any header or row name) with rows as independent variables' names.

**dependent_parameters:** *{string, required}* 
> Directory of a text file containing the name (>1 dependent parametera is not yet supported) of independent variable. Data must be stored in one column and one row (one entery, without any header or row name) containing the name of dependent variable.

**normalize:** *{bool, Default=False}*
> `True` is recommended. If `True`, all variables (dependent and independent) are nornalized to have unit variance and zero mean (i.e. Standardization). Normalization typically improves performance.

**test_size:** *{float, required}*
> Fraction of data (0<=test_size<=1) in each group to be randomly selected and excluded from training. Useful for testing ML models on holdout data.

**split_random_state:** *{integer, optional}*
> Seed number for random selection of test data.

**cache:** *{string, optional}*
> Directory for cache. If specified, data will be read from disk and model training is performed in batches. This can adversely impact model training overall CIMLA performance. Useful when data is too large to fit in memory. 

**sample_column:** *{string, optional}*
> Name of the column containing row names in the observational data of both groups. Leave it empty if this column is not labled in data.

### ML:
**type:** *{(RF , MLP , XGB), required}*
> Type of the ML model to be trained in each group. Tree-based models (RF,XGB) are trained by cross-validation on train split to select the best hyper-patameters among the various user-defined settings. 

**task:** *{(regression), required}*
> The ML task (i.e. loss) to train in each group. `classification` is not yet supported.

<br>

#### Tree (RF and XGBoost) Specific Parameters:
**max_depth:** *{list(integer), required}*
> The maximum depth of the decision trees. A list of integer(s) containing numbers to be evaluated in cross-validation during training. 

**scoring:** *{(rmse , neg_mean_squared_error), required}*
> scoring function to be used during cross-validation for hyper-parameter tuning. Use `rmse` for XGB regression and `neg_mean_squared_error` for RF regression.

<br>

#### RF Specific Parameters:
**n_estimators:** *{list(integer), required}*
> The number of decision trees in the forest. A list of integer(s) containing numbers to be evaluated in cross-validation during training. 

**max_features:** *{list(int , float , "sqrt" , "log2"), required}*
> The number of features to consider when looking for the best split. TODO

**min_samples_leaf:** *{list(integer), required}*
> The minimum number of samples required to be at a leaf node. A list of integer(s) containing numbers to be evaluated in cross-validation during training.

**max_leaf_nodes:** *{list(integer), required}*
> The maximum number of leaf nodes a tree can attain during training (controls tree size). A list of integer(s) containing numbers to be evaluated in cross-validation during training.

**cv:** *{integer, required}*
> Number of folds to be used for cross-validation during training. 

<br>

#### XGBoost Specific Parameters

**min_child_weight:** *{list(integer), required}*
> The maximum number of leaf nodes a tree can attain during training (controls tree size). A list of integer(s) containing numbers to be evaluated in cross-validation during training.




```yaml
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





