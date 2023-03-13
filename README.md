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
**path_g1:** *string, required* 
> Directory of the observational data for group 1. Data must be stored in .csv format with rows as samples and columns as variables, also header (column names) and row names are required. 

**path_g2:** *string, required* 
> Directory of the observational data for group 2. Data must be stored in .csv format with rows as samples and columns as variables, also header (column names) and row names are required. 

**number_samples_g1:** *integer, required* 
> Number of samples (rows) in the observational data of group 1.

**number_samples_g2:** *integer, required* 
> Number of samples (rows) in the observational data of group 2.

**independent_parameters:** *string, required* 
> Directory of a text file containing the names of independent variables. Data must be stored in one column (without any header or row name) with rows as independent variables' names.

**dependent_parameters:** *string, required* 
> Directory of a text file containing the name (>1 dependent parametera is not yet supported) of independent variable. Data must be stored in one column and one row (one entery, without any header or row name) containing the name of dependent variable.

**normalize:** *bool, Default=False*
> `True` is recommended. If `True`, all variables (dependent and independent) are nornalized to have unit variance and zero mean (i.e. Standardization). Normalization typically improves performance.

**test_size:** *float, required*
> Fraction of data (0<=test_size<=1) in each group to be randomly selected and excluded from training. Useful for testing ML models on holdout data.

**split_random_state:** *integer, optional*
> Seed number for random selection of test data.

**cache:** *string, optional*
> Directory for cache. If specified, data will be read from disk and model training is performed in batches. This can adversely impact model training overall CIMLA performance. Useful when data is too large to fit in memory. 

**sample_column:** *string, optional*
> Name of the column containing row names in the observational data of both groups. Leave it empty if this column is not labled in data.

### ML:
**type:** *{RF , MLP , XGB}, required*
> Type of the ML model to be trained in each group. Tree-based models (RF,XGB) are trained by cross-validation on train split to select the best hyper-patameters among the various user-defined settings. 

**task:** *(regression), required*
> The ML task (i.e. loss) to train in each group. `classification` is not yet supported.

<br>

#### Tree (RF and XGBoost) Specific Parameters:
**max_depth:** *list(integer), required*
> The maximum depth of the decision trees. A list of integer(s) containing numbers to be evaluated in cross-validation during training. 

**scoring:** *{rmse , neg_mean_squared_error}, required*
> scoring function to be used during cross-validation for hyper-parameter tuning. Use `rmse` for XGB regression and `neg_mean_squared_error` for RF regression.

<br>

#### RF Specific Parameters:
**n_estimators:** *list(integer), required*
> The number of decision trees in the forest. A list of integer(s) containing numbers to be evaluated in cross-validation during training. 

**max_features:** *list(int , float , "sqrt" , "log2"), required*
> The number of features to consider when looking for the best split. TODO

**min_samples_leaf:** *list(integer), required*
> The minimum number of samples required to be at a leaf node. A list of integer(s) containing numbers to be evaluated in cross-validation during training.

**max_leaf_nodes:** *list(integer), required*
> The maximum number of leaf nodes a tree can attain during training (controls tree size). A list of integer(s) containing numbers to be evaluated in cross-validation during training.

**cv:** *integer, required*
> Number of folds to be used for cross-validation during training. 

<br>

#### XGBoost Specific Parameters

**min_child_weight:** *list(integer), required*
> Minimum sum of instance weight (hessian) needed in a child. See XGBoost documentation for more details. A list of integer(s) containing numbers to be evaluated in cross-validation during training.

**subsample:** *list(float), required*
> Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. See XGBoost documentation for more details. A list of float(s) containing numbers to be evaluated in cross-validation during training.

**colsample:** *list(float), required*
> Controls `colsample_bynode` parameter of XGBoost. Subsample ratio of columns for each node (split). See XGBoost documentation for more details. A list of float(s) containing numbers to be evaluated in cross-validation during training.

**eta:** *list(float), required*
> Step size shrinkage used in update to prevents overfitting. See XGBoost documentation for more details. A list of float(s) containing numbers to be evaluated in cross-validation during training.

**l2:** *list(float), required*
> Controls `lambda` parameter of XGBoost. L2 regularization term on weights. See XGBoost documentation for more details. A list of float(s) containing numbers to be evaluated in cross-validation during training.

**l1:** *list(float), required*
> Controls `alpha` parameter of XGBoost. L1 regularization term on weights. See XGBoost documentation for more details. A list of float(s) containing numbers to be evaluated in cross-validation during training.

**max_boost_round:** *integer, required*
> Controls `num_boost_round` parameter of XGBoost. Maximum number of boosting iterations. See XGBoost documentation for more details.

**early_stop_round:** *integer, required*
> Cross-Validation metric (average of validation metric computed over CV folds) needs to improve at least once in every `early_stop_rounds` round(s) to continue training. See XGBoost documentation for more details.

**early_stop_tol:** *float, required*
> Minimum absolute change in score to be qualified as an improvement. See XGBoost documentation for more details.TODO

<br>

#### MLP Specific Parameters

**hidden_channels:** *list(integer), required*
> A list of size *h* that determines the size of *h* hidden layers. The left most element corresponds to the layer right after the input, and the right most element correponds to the the layer before the output.

**dropout:** *float, optional*
> If specified, a dropout layer with `prob=dropout` is used after the input layer. (0<dropout<=1)

**dense_layers_l2:** *float, optional*
> If specified, l2 regularization term with the specified weight is applied to weights of each of the hidden layers.

<br>

### attribution:
**type:** *{tree_shap , deep_shap}, required*
> Type of the attribution model to be used (must match with the type of the ML model). Currently, support tree_shap for tree-based ML models (RF and XGBoost) and deep-shap for MLP.

**data_split:** *{train , test}, required*
> Split of data to be used for interpretation of the trained models.

**data_group:** *{1 , 2}, required*
> The group of data to be used for interpretation of the trained models.

**data_size:** *float, required*
> Fraction of data (0<data_size<=1) to be used for interpretation of the trained models. The specified fraction is randomly sampled from split = `data_split` of group = `data_group`.

**global_type:** *{rmsd}, required*
> Function used for aggregating local scores into global scores. For now only support `rmsd` (Root-Mean-Square-Difference).

<br>

### post_process:
**local_scores_path:** *string, optional*
> If specified, local attribution scores for both models are saved in specified directiory.

**global_scores_path:** *string, optional*
> If specified, global CIMLA scores for both models are saved in specified directiory.

**ML_performance_save_path:** *string, optional*
> If specified, performance of both ML models on both train and test splits are saved in specified directiory.

**ML_performance_metric** *{r2 , accuracy , mse}, optional*
> Required if `ML_performance_save_path` is specified.


