import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()#This is for compatibility of shap and tf, might be able to remove in future
from CIMLA.ML.BaseModels import baseML
from CIMLA.ML.archs.MLP import build_MLP
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
                 scoring = 'accuracy',
                 n_jobs = 1):
        super(RF,self).__init__()

        pip = Pipeline(steps = [('rf', RandomForestClassifier(n_jobs = n_jobs, class_weight = 'balanced'))])
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

        print(params)

        return params

    def _build_model(self, pars, nSample, cache = None, batch_size = 3000):
        if cache:
            warm_start = True
            n_batch = np.ceil(nSample/batch_size)#this is an approximate, but best we can do!
            n_estimators = int(max(1, pars['n_estimators']//n_batch))
        else:
            warm_start = False
            n_estimators = pars['n_estimators']

        model = RandomForestClassifier(n_estimators = n_estimators,
                                      max_depth = pars['max_depth'],
                                      max_features = pars['max_features'],
                                      min_samples_leaf = pars['min_samples_leaf'],
                                      max_leaf_nodes = pars['max_leaf_nodes'],
                                      n_jobs = self.n_jobs_,
                                      warm_start = warm_start,
                                      class_weight = 'balanced')
        return model, n_estimators

    def _batch_training(self, data, model, n_estimators):
        for b in data.batch(split = 'train', size = 3000, shuffle = True):
            curr = b.numpy()
            model.fit(X = curr[:,:-1], y = curr[:,-1].reshape(-1,))
            model.n_estimators += n_estimators
        model.n_estimators -= n_estimators
        return model

    def train(self, data):
        # first do cv on <=3000 samples to select best parameters
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
        self.model.compile(optimizer = "Adam", loss = tf.keras.losses.BinaryCrossentropy(from_logits = True))
        self.classW_ = class_weight

    def _tarin_on_batch(self, batch_x, batch_y):
        self.model.train_on_batch(x = batch_x, y = batch_y, reset_metrics = False, class_weight = self.classW_)

    def train(self, data, batch_size = 128, epochs = 100, shuffle = True):
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

    def predict(self, x, tresh = 0.5):
        logit = self.model.predict_on_batch(x = x)
        p = np.exp(logit) / (1 + np.exp(logit))
        #p = self.model.predict_on_batch(x = x)

        if tresh:
            return p >= tresh
        else:
            return p

    def save(self, path):
        #TODO
        pass

####TOdos
# if possible train with logit but run shap on prob, or modify code ...
#
