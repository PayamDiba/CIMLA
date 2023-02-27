import yaml
import CIMLA.defaults as defaults
import pandas as pd
import numpy as np
import dask.dataframe as dd
from CIMLA.Dataset import data
from CIMLA.ML.Regression import xgboost
#from ML.Classification import RF_clf
from CIMLA.attribution.SHAP import treeSHAP, deepSHAP
from CIMLA.CIMLA import cimla
from CIMLA.PostProcessor import processor

"""
TODOs:
1. write a checker method that makes sure the provided flags are legit
"""

class parser(object):

    @staticmethod
    def parse(config_file):
        with open(config_file,'r') as cf:
            config = yaml.safe_load(cf)

        config = defaults.params(config)
        data_g1, data_g2, n_features,classW1, classW2 = parser._parse_data(config['data'], task = config['ML']['task'])
        ML_g1, ML_g2 = parser._parse_ML(config['ML'], n_features, class_weights = (classW1, classW2))
        att_g1, att_g2 = parser._parse_attribution(config['attribution'])
        pipeline = cimla(data_g1 = data_g1,
                         data_g2 = data_g2,
                         ML_g1 = ML_g1,
                         ML_g2 = ML_g2,
                         attr_g1 = att_g1,
                         attr_g2 = att_g2)
        proc = parser._parse_postproc(config['post_process'], pipeline)
        return pipeline, proc, config

    @staticmethod
    def _to_classification(df1,df2,y_name, type='median'):

        t = df1[y_name].values.flatten().tolist() + df2[y_name].values.flatten().tolist()
        t = np.median(t)

        new = []
        for _,r in df1.iterrows():
            if r[y_name] > t:
                new.append(1)
            else:
                new.append(0)
        df1[y_name] = new
        pos = sum(new)
        neg = len(new) - pos
        total = len(new)
        weights_1 = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

        new = []
        for _,r in df2.iterrows():
            if r[y_name] > t:
                new.append(1)
            else:
                new.append(0)
        df2[y_name] = new
        pos = sum(new)
        neg = len(new) - pos
        total = len(new)
        weights_2 = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

        return df1,df2, weights_1, weights_2

    @staticmethod
    def _parse_data(data_config, task = 'regression'):
        if data_config['cache']:#use dask
            index_col = data_config['sample_column'] if data_config['sample_column'] else 'Unnamed: 0'
            df1 = dd.read_csv(data_config['path_g1'], header = 0, dtype = {index_col : 'string'})#.set_index(index_col)
            df2 = dd.read_csv(data_config['path_g2'], header = 0, dtype = {index_col : 'string'})#.set_index(index_col)
        else:
            df1 = pd.read_csv(data_config['path_g1'], header = 0, index_col = 0)
            df2 = pd.read_csv(data_config['path_g2'], header = 0, index_col = 0)


        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)

        indep = pd.read_csv(data_config['independent_parameters'], header = None, index_col = None, dtype = 'string').values.flatten().tolist()
        dep = pd.read_csv(data_config['dependent_parameters'], header = None, index_col = None, dtype = 'string').values.flatten().tolist()

        weights_1 = weights_2 = None
        if task == 'classification':
            df1,df2,weights_1, weights_2 = parser._to_classification(df1, df2, dep[0], type='median')


        data_g1 = data(data = df1,
                       nSample = data_config['number_samples_g1'],
                       indep_params = indep.copy(),
                       dep_params = dep,
                       normalize = data_config['normalize'],
                       test_size = data_config['test_size'],
                       split_rnd_state = data_config['split_random_state'],
                       cache = data_config['cache'],
                       task = task,
                       label = 'g1')

        data_g2 = data(data = df2,
                       nSample = data_config['number_samples_g2'],
                       indep_params = indep.copy(),
                       dep_params = dep,
                       normalize = data_config['normalize'],
                       test_size = data_config['test_size'],
                       split_rnd_state = data_config['split_random_state'],
                       cache = data_config['cache'],
                       task = task,
                       label = 'g2')

        return data_g1, data_g2, len(indep), weights_1, weights_2

    @staticmethod
    def _parse_ML(ML_config, n_features, class_weights = None):
        if ML_config['type'] == 'RF':
            if ML_config['task'] == 'regression':
                from ML.Regression import RF
            else:
                from ML.Classification import RF

            max_depth = [i for i in ML_config['max_depth'] if i != 'None']
            if 'None' in ML_config['max_depth']:
                max_depth.append(None)

            max_features = [i for i in ML_config['max_features'] if i != 'None']
            if 'None' in ML_config['max_features']:
                max_features.append(None)


            model1 = RF(n_estimators = ML_config['n_estimators'],
                        max_depth = ML_config['max_depth'],
                        max_features = ML_config['max_features'],
                        min_samples_leaf = ML_config['min_samples_leaf'],
                        max_leaf_nodes = ML_config['max_leaf_nodes'],
                        cv = ML_config['cv'],
                        scoring = ML_config['scoring'],
                        n_jobs = 1)

            model2 = RF(n_estimators = ML_config['n_estimators'],
                        max_depth = ML_config['max_depth'],
                        max_features = ML_config['max_features'],
                        min_samples_leaf = ML_config['min_samples_leaf'],
                        max_leaf_nodes = ML_config['max_leaf_nodes'],
                        cv = ML_config['cv'],
                        scoring = ML_config['scoring'],
                        n_jobs = 1)

        elif ML_config['type'] == 'MLP':
            if ML_config['task'] == 'regression':
                from CIMLA.ML.Regression import deepMLP
            else:
                from CIMLA.ML.Classification import deepMLP

            model1 = deepMLP(input_shape = n_features, mid_channels = ML_config['hidden_channels'], l2 = ML_config['dense_layers_l2'], dropout = ML_config['dropout'], class_weight = class_weights[0])
            model2 = deepMLP(input_shape = n_features, mid_channels = ML_config['hidden_channels'], l2 = ML_config['dense_layers_l2'],dropout = ML_config['dropout'], class_weight = class_weights[1])
        elif ML_config['type'] == 'XGB':
            model1 = xgboost(max_depth = ML_config['max_depth'],
                             min_child_weight = ML_config['min_child_weight'],
                             subsample = ML_config['subsample'],
                             colsample_bynode = ML_config['colsample'],
                             eta = ML_config['eta'],
                             l2 = ML_config['l2'],
                             l1 = ML_config['l1'],
                             max_boost_round = ML_config['max_boost_round'],
                             early_stopping_round = ML_config['early_stop_round'],
                             early_stopping_tol = ML_config['early_stop_tol'],
                             scoring = ML_config['scoring'],
                             objective = 'reg:squarederror')

            model2 = xgboost(max_depth = ML_config['max_depth'],
                             min_child_weight = ML_config['min_child_weight'],
                             subsample = ML_config['subsample'],
                             colsample_bynode = ML_config['colsample'],
                             eta = ML_config['eta'],
                             l2 = ML_config['l2'],
                             l1 = ML_config['l1'],
                             max_boost_round = ML_config['max_boost_round'],
                             early_stopping_round = ML_config['early_stop_round'],
                             early_stopping_tol = ML_config['early_stop_tol'],
                             scoring = ML_config['scoring'],
                             objective = 'reg:squarederror')
        else:
            raise ValueError('{} machine learning model is not yet supported.'.format(ML_config['type']))

        return model1, model2

    @staticmethod
    def _parse_attribution(att_config):
        if att_config['type'] == 'tree_shap':
            model1 = treeSHAP()
            model2 = treeSHAP()
        elif att_config['type'] == 'deep_shap':
            model1 = deepSHAP()
            model2 = deepSHAP()
        else:
            raise ValueError('"{}" attribution model is not supported.'.format(att_config['type']))

        return model1, model2

    @staticmethod
    def _parse_postproc(proc_config, pipeline):
        proc = processor(CIMLA = pipeline,
                         save_local = proc_config['local_scores_path'],
                         save_global = proc_config['global_scores_path'],
                         save_ML = proc_config['ML_save_path'],
                         save_performance = proc_config['ML_performance_save_path'],
                         metric = proc_config['ML_performance_metric'])
        return proc
