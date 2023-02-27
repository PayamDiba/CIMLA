import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()#This is for compatibility of shap and tf, might be able to remove in future
from CIMLA.ML.BaseModels import baseML
from CIMLA.ML.archs.MLP import build_MLP
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import xgboost as xgb
import copy


class RF(baseML):
    def __init__(self,
                 n_estimators,
                 max_depth,
                 max_features,
                 min_samples_leaf,
                 max_leaf_nodes,
                 cv,
                 scoring = 'neg_mean_squared_error',
                 n_jobs = 1):
        super(RF,self).__init__()

        pip = Pipeline(steps = [('rf', RandomForestRegressor(n_jobs = n_jobs))])
        params = {'rf__n_estimators': n_estimators,
                  'rf__max_depth': max_depth,
                  'rf__max_features': max_features,
                  'rf__min_samples_leaf': min_samples_leaf,
                  'rf__max_leaf_nodes': max_leaf_nodes
                  }

        self.gscv_ = GridSearchCV(pip, param_grid = params, cv = cv, refit = False, scoring = scoring)
        self.n_jobs_ = n_jobs
        self.model = None

    def _CV(self, sampled_date):
        """
        sampled_date is must be pandas df
        """
        x = sampled_date.iloc[:,:-1].values
        y = sampled_date.iloc[:,-1].values.reshape(-1)
        self.gscv_.fit(x, y)
        pars = self.gscv_.best_params_
        params = {'n_estimators': pars['rf__n_estimators'],
                  'max_depth': pars['rf__max_depth'],
                  'max_features': pars['rf__max_features'],
                  'min_samples_leaf': pars['rf__min_samples_leaf'],
                  'max_leaf_nodes': pars['rf__max_leaf_nodes']}

        print("    -- final model: ", params)

        return params

    def _build_model(self, pars, nSample, cache = None, batch_size = 3000):
        if cache:
            warm_start = True
            n_batch = np.ceil(nSample/batch_size)#this is an approximate, but best we can do!
            n_estimators = int(max(1, pars['n_estimators']//n_batch))
        else:
            warm_start = False
            n_estimators = pars['n_estimators']

        model = RandomForestRegressor(n_estimators = n_estimators,
                                      max_depth = pars['max_depth'],
                                      max_features = pars['max_features'],
                                      min_samples_leaf = pars['min_samples_leaf'],
                                      max_leaf_nodes = pars['max_leaf_nodes'],
                                      n_jobs = self.n_jobs_,
                                      warm_start = warm_start)
        return model, n_estimators

    def _batch_training(self, data, model, n_estimators):
        for b in data.batch(split = 'train', size = 6000, shuffle = True):
            curr = b.numpy()
            model.fit(X = curr[:,:-1], y = curr[:,-1].reshape(-1,))
            model.n_estimators += n_estimators
        model.n_estimators -= n_estimators
        return model

    def train(self, data):
        # first do cv on <=6000 samples to select best parameters
        sampled_date = data.sample(n = 6000, split = 'train')
        if data.cache_: #data is dask dataframe, should convert it to pandas
            sampled_date = sampled_date.compute()
        best_pars = self._CV(sampled_date)

        # train the final model
        model, n_estimators = self._build_model(pars = best_pars, nSample = data.nSample['train'], cache = data.cache_, batch_size = 6000)
        if data.cache_: #batch training is needed
            self.model = self._batch_training(data, model, n_estimators)
        else: #normal training, data is pandas dataframe
            train_data = data.getDatadf(split = 'train')
            model.fit(X = train_data.iloc[:,:-1], y = train_data.iloc[:,-1])
            self.model = model

    def predict(self, x):
        if not self.model:
            raise ValueError('model has not yet been trained.')
        return self.model.predict(x)

    def save(self, path):
        if not self.model:
            raise ValueError('model has not yet been trained.')
        joblib.dump(self.model, path)




class deepMLP(baseML):
    def __init__(self, input_shape, mid_channels, l2 = 0, dropout = None, class_weight = None):
        super(deepMLP,self).__init__()
        self.model = build_MLP(input_shape, mid_channels, l2 = l2, dropout = dropout)
        self.model.compile(optimizer = "Adam", loss = "mse")
        self.classW_ = class_weight

    def _tarin_on_batch(self, batch_x, batch_y):
        self.model.train_on_batch(x = batch_x, y = batch_y, reset_metrics = False, class_weight = self.classW_)

    def train(self, data, batch_size = 128, epochs = 100, shuffle = True, class_weight = None):
        if data.cache_:
            batches = data.batch(split = 'train', size = batch_size, shuffle = shuffle)
            for _ in range(epochs):
                for b in batches:
                    self._tarin_on_batch(b[:,:-1], b[:,-1])
        else:
            df = data.getDatadf(split = 'train')
            X = df.values[:,:-1]
            y = df.values[:,-1]
            self.model.fit(x = X, y = y, batch_size = batch_size, epochs = epochs, shuffle = shuffle, verbose = 0, class_weight = self.classW_)

    def predict(self, x):
        return self.model.predict_on_batch(x = x)

    def save(self, path):
        #TODO
        pass


class xgboost(baseML):
    def __init__(self,
                 max_depth,
                 min_child_weight,
                 subsample,
                 colsample_bynode,
                 eta,
                 l2,
                 l1,
                 max_boost_round = 250,
                 early_stopping_round = 30,#must be specified ## TODO
                 early_stopping_tol = 1e-2,
                 scoring = 'mae',
                 objective = 'reg:squarederror'):
        super(xgboost,self).__init__()

        self.grid_param_ = {
            'max_depth':max_depth,
            'min_child_weight': min_child_weight,
            'eta': eta,
            'lambda': l2,
            'alpha' : l1,
            'subsample': subsample,
            'colsample_bynode': colsample_bynode,
            'objective':objective,
            'eval_metric':scoring,
        }

        self.max_round_ = max_boost_round
        self.early_stopper_ = xgb.callback.EarlyStopping(rounds = early_stopping_round, min_delta = early_stopping_tol)

    def _get_param_dict(self):
        ret = []
        for md in self.grid_param_['max_depth']:
            for mc in self.grid_param_['min_child_weight']:
                for s in self.grid_param_['subsample']:
                    for cs in self.grid_param_['colsample_bynode']:
                        for et in self.grid_param_['eta']:
                            for l in self.grid_param_['lambda']:
                                for l1 in self.grid_param_['alpha']:
                                    curr = {
                                        'max_depth':md,
                                        'min_child_weight': mc,
                                        'eta': et,
                                        'lambda': l,
                                        'alpha': l1,
                                        'subsample': s,
                                        'colsample_bynode': cs,
                                        'objective':self.grid_param_['objective'],
                                        'eval_metric':self.grid_param_['eval_metric'],
                                    }
                                    ret += [curr]
        return ret

    def _CV(self, sampled_data, seed = 1234):
        x = sampled_data.iloc[:,:-1].values
        y = sampled_data.iloc[:,-1].values
        data_train = xgb.DMatrix(x, label = y)
        best_score = 1e8
        best_param = None
        max_round = None
        allParDicts = self._get_param_dict()
        for params in allParDicts:
            cv_results = xgb.cv(
                params = params,
                dtrain = data_train,
                num_boost_round = self.max_round_,
                seed = seed,
                nfold = 3,
                metrics = params['eval_metric'],
                callbacks = [copy.deepcopy(self.early_stopper_)]
            )
            mean_score = cv_results['test-{}-mean'.format(params['eval_metric'])].min()
            if mean_score < best_score:
                best_score = mean_score
                best_param = params.copy()
                max_round = cv_results['test-{}-mean'.format(params['eval_metric'])].argmin() + 1

        print("    -- final model has {} trees: ".format(max_round), best_param)

        return best_param, max_round

    def _batch_training(self, params, data, n_round):
        last_model = None
        for b in data.batch(split = 'train', size = 3000, shuffle = True):
            curr = b.numpy()
            x = curr[:,:-1]
            y = curr[:,-1]
            data_train = xgb.DMatrix(x, label = y)
            model = xgb.train(
                params = params,
                dtrain = data_train,
                num_boost_round = n_round,
                xgb_model = last_model
            )
            last_model = model

        return model

    def train(self, data, nsample_cv = 6000):

        # first do cv to select best parameters and num_boost_round (CV is always required to get num_boost_round)
        sampled_date = data.sample(n = nsample_cv, split = 'train')
        if data.cache_: #data is dask dataframe, should convert it to pandas
            sampled_date = sampled_date.compute()
        best_pars, n_round = self._CV(sampled_date)

        # train the final model
        if data.cache_: #batch training is needed
            self.model = self._batch_training(best_pars, data, n_round)
        else: #normal training, data is pandas dataframe
            train_data = data.getDatadf(split = 'train')
            x = train_data.iloc[:,:-1].values
            y = train_data.iloc[:,-1].values
            data_train = xgb.DMatrix(x, label = y)
            model = xgb.train(
                params = best_pars,
                dtrain = data_train,
                num_boost_round = n_round,
            )

            self.model = model

    def predict(self, x):
        if not self.model:
            raise ValueError('model has not yet been trained.')
        data = xgb.DMatrix(x)
        return self.model.predict(data)

    def save(self, path):
        if not self.model:
            raise ValueError('model has not yet been trained.')
        joblib.dump(self.model, path)
