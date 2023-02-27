import pandas as pd
import numpy as np
from CIMLA.attribution.metrics import RMSD
import joblib
from CIMLA.utils import mkdir, runningMean
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import shutil


class processor(object):
    def __init__(self,
                 CIMLA,
                 save_local = None,
                 save_global = None,
                 save_ML = None,
                 save_performance = None,
                 metric = 'mse'):


        self.cimla_ = CIMLA
        self.local_path_ = save_local
        self.global_path_ = save_global
        self.ml_path_ = save_ML
        self.perf_path_ = save_performance
        self.metric_ = metric

    def _save_local(self, path):
        g1 = self.cimla_.attr_['g1'].local_feature_importance
        g2 = self.cimla_.attr_['g2'].local_feature_importance
        g1 = pd.DataFrame(g1)
        g2 = pd.DataFrame(g2)
        g1.columns = self.cimla_.data_['g1'].indep_pars_
        g2.columns = self.cimla_.data_['g2'].indep_pars_
        g1.to_csv(path + '/local_feature_importance_g1.csv', header = True, index = True)
        g2.to_csv(path + '/local_feature_importance_g2.csv', header = True, index = True)


    def _save_global(self,path):
        scores = self.cimla_.feature_importance.reshape(1,-1)
        scores = pd.DataFrame(scores)
        scores.columns = self.cimla_.data_['g1'].indep_pars_
        scores.to_csv(path + '/global_feature_importance.csv', header = True, index = True)


    def _save_ml(self, path):
        ml1 = self.cimla_.ml_['g1'].model
        ml2 = self.cimla_.ml_['g2'].model
        joblib.dump(ml1, path + '/ML_g1')
        joblib.dump(ml2, path + '/ML_g2')

    def _compute_perf_on_group_and_split(self, g, split, function):
        if self.cimla_.data_[g].cache_: #use batch
            perf = runningMean()
            batches = self.cimla_.data_[g].batch(split = split, size = 128, shuffle = False, sample_n = None)
            y_pred = []
            y_true = []
            for b in batches:
                currB = b.numpy()
                X = currB[:,:-1]
                y = currB[:,-1]
                y_hat = self.cimla_.ml_[g].predict(X)
                currPerf = function(y_hat, y)
                currSize = X.shape[0]
                perf.append(currPerf, currSize)
                y_pred.append(y_hat.flatten())
                y_true.append(y.flatten())
            return perf.mean, np.concatenate(y_pred).reshape(-1,1), np.concatenate(y_true).reshape(-1,1)
            ## TODO: make sure the returned y_pred has the right shape

        else:
            data = self.cimla_.data_[g].getDatadf(split = split).values
            X = data[:,:-1]
            y = data[:,-1]
            y_hat = self.cimla_.ml_[g].predict(X)
            return function(y, y_hat),y_hat.reshape(-1,1), y.reshape(-1,1)


    def _compute_perf(self, function):
        train_g1, yP_g1_train, yT_g1_train = self._compute_perf_on_group_and_split(g = 'g1', split = 'train', function = function)
        train_g2, yP_g2_train, yT_g2_train = self._compute_perf_on_group_and_split(g = 'g2', split = 'train', function = function)
        test_g1, yP_g1_test, yT_g1_test = self._compute_perf_on_group_and_split(g = 'g1', split = 'test', function = function)
        test_g2, yP_g2_test, yT_g2_test = self._compute_perf_on_group_and_split(g = 'g2', split = 'test', function = function)

        perf = np.array([train_g1, test_g1, train_g2, test_g2]).reshape(1,-1)
        perf = pd.DataFrame(perf)
        perf.columns = ['train_g1', 'test_g1', 'train_g2', 'test_g2']

        y_g1_train = pd.DataFrame(np.concatenate([yT_g1_train, yP_g1_train], axis = 1))
        y_g2_train = pd.DataFrame(np.concatenate([yT_g2_train, yP_g2_train], axis = 1))
        y_g1_test = pd.DataFrame(np.concatenate([yT_g1_test, yP_g1_test], axis = 1))
        y_g2_test = pd.DataFrame(np.concatenate([yT_g2_test, yP_g2_test], axis = 1))

        y_g1_train.columns = ['y_true','y_pred']
        y_g2_train.columns = ['y_true','y_pred']
        y_g1_test.columns = ['y_true','y_pred']
        y_g2_test.columns = ['y_true','y_pred']

        return perf, y_g1_train, y_g2_train, y_g1_test, y_g2_test

    def _save_perf(self, path):
        if self.metric_ == 'r2':
            func = r2_score
        elif self.metric_ == 'accuracy':
            func = accuracy_score
        elif self.metric_ == 'mse':
            func = mean_squared_error
        else:
            raise ValueError('ML evaluation metric {} is not implemented.'.format(self.metric_))

        perf, y_g1_train, y_g2_train, y_g1_test, y_g2_test = self._compute_perf(func)
        perf.to_csv(path + '/ML_performance.csv', header = True, index = True)

        y_g1_train.to_csv(path + '/y_g1_train.csv', header = True, index = True)
        y_g2_train.to_csv(path + '/y_g2_train.csv', header = True, index = True)
        y_g1_test.to_csv(path + '/y_g1_test.csv', header = True, index = True)
        y_g2_test.to_csv(path + '/y_g2_test.csv', header = True, index = True)

    def process(self):
        if self.local_path_:
            mkdir(self.local_path_)
            self._save_local(self.local_path_)
        if self.global_path_:
            mkdir(self.global_path_)
            self._save_global(self.global_path_)
        if self.ml_path_:
            mkdir(self.ml_path_)
            self._save_ml(self.ml_path_)
        if self.perf_path_:
            mkdir(self.perf_path_)
            self._save_perf(self.perf_path_)

    def clear(self):
        if self.cimla_.data_['g1'].cache_:
            shutil.rmtree(self.cimla_.data_['g1'].cache_)
