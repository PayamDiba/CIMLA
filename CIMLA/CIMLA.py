import numpy as np
from attribution.SHAP import deepSHAP, treeSHAP
from attribution.metrics import RMSD, MSD
from utils import runningMean
class cimla (object):
    def __init__(self,
                 data_g1,
                 data_g2,
                 ML_g1,
                 ML_g2,
                 attr_g1,
                 attr_g2):

        """
        attr_data: data split used for explanation: 'train', 'test'
        """

        self.data_ = {'g1': data_g1,
                      'g2': data_g2}

        self.ml_ = {'g1': ML_g1,
                    'g2': ML_g2}

        self.attr_ = {'g1': attr_g1,
                     'g2': attr_g2}

        self.feature_importance = None
        self.task = data_g1.task
        if self.task == 'regression':
            self.ml_out_ = 'raw'
        else:
            self.ml_out_ = 'raw'

    def run(self, attr_data_split = 'train', attr_data_group = 2, attr_data_size = 1, global_type = 'rmsd'):
        dg = 'g' + str(attr_data_group)
        sample_size = self.data_[dg].nSample[attr_data_split] * attr_data_size if attr_data_size < 1 else None
        attr_data = self._get_attr_data(data = self.data_[dg], split = attr_data_split, sample_size = sample_size)
        background_x = self._get_background_data(data = self.data_[dg], split = attr_data_split, backgroud_size = 1000)# if isinstance(self.attr_['g1'], deepSHAP) else None

        print("* Trainig ML Models ...")
        print("    -- training ML model for group 1 ...")
        self._trainML(self.ml_['g1'], self.data_['g1'])
        print("    -- training ML model for group 2 ...")
        self._trainML(self.ml_['g2'], self.data_['g2'])


        print("* Running Attribution Model ...")
        for g in ['g1','g2']:
            self.attr_[g].set_explainer(self.ml_[g], background_x, model_output = self.ml_out_)
            #print(self.attr_[g].explainer_.data.shape[0])
        self.feature_importance = self._runAM(attr_data, global_type)


    def _trainML(self, model, data):
        model.train(data)

    def _get_attr_data(self, data, split, sample_size = None):
        if data.cache_:
            return data.batch(split, 1024, shuffle = True, sample_n = sample_size)
        elif sample_size:
            return data.sample(n = sample_size, split = split).iloc[:,:-1].values
        else:
            return data.getDatadf(split).iloc[:,:-1].values


    def _get_background_data(self, data, split, backgroud_size):
        bg = data.sample(backgroud_size, split).iloc[:,:-1]
        if data.cache_:
            bg = bg.compute()
        return bg.values

    def _compute_global(self, type, feat_importance_g1, feat_importance_g2):
        if type == 'rmsd':
            func = RMSD
        elif type == 'msd':
            func = MSD
        else:
            raise ValueError('global attribution score {} is not implemented.'.format(type))

        return func(s1 = feat_importance_g1, s2 = feat_importance_g2)

    def _runAM(self, attr_data, global_type, store_local = False):
        if isinstance(attr_data, np.ndarray):
            g1 = self.attr_['g1'].explain(attr_data)
            g2 = self.attr_['g2'].explain(attr_data)
            if isinstance(g1,list):
                g1 = g1[-1] #contribution to the probability of belonging to the last group
                g2 = g2[-1] #contribution to the probability of belonging to the last group

            if store_local:
                self.attr_['g1'].append_local_importance(g1)
                self.attr_['g2'].append_local_importance(g2)
            return self._compute_global(global_type, g1, g2)

        else: #data is tensorflow dataset with batches
            score = runningMean()
            global_type = 'msd' if global_type == 'rmsd' else global_type
            for b in attr_data:
                currX = b.numpy()[:,:-1]
                g1 = self.attr_['g1'].explain(currX)
                g2 = self.attr_['g2'].explain(currX)
                if isinstance(g1,list):
                    g1 = g1[-1] #contribution to the probability of belonging to the last group
                    g2 = g2[-1] #contribution to the probability of belonging to the last group

                if store_local:
                    self.attr_['g1'].append_local_importance(g1)
                    self.attr_['g2'].append_local_importance(g2)

                curr = self._compute_global(global_type, g1, g2)
                size = currX.shape[0]
                score.append(curr, size)
                if global_type == 'msd':#return rmsd
                    ret = np.power(score.mean, 0.5)
            return ret
