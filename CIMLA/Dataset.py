import pandas as pd
import numpy as np
import sklearn.preprocessing
import dask_ml.preprocessing
import sklearn.model_selection
import dask_ml.model_selection
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  #only display errors
import tensorflow as tf
import h5py
from utils import mkdir, generate_from_hdf5
#tf.compat.v1.disable_v2_behavior()#This is for compatibility of shap and tf, might be able to remove in future

class data(object):
    def __init__(self,
                 data,
                 nSample,
                 indep_params,
                 dep_params,
                 normalize,
                 test_size,
                 split_rnd_state = None,
                 cache = None,
                 task = 'regression',
                 label = 'g1'):

        assert(isinstance(indep_params,list))
        assert(isinstance(dep_params,list))
        #For now only support single dependent parameter
        assert(len(dep_params) == 1)
        self.task = task
        self.label_ = label
        self.cache_ = cache
        self.indep_pars_ = indep_params
        self.dep_pars_ = dep_params
        df = self._sort_xy(data, nSample, normalize, task)
        self.data_ = self._split(df = df, test_size = test_size, random_state = split_rnd_state)
        self.nSample = {'train': nSample * (1-test_size), 'test': nSample * test_size}
        if cache:
            mkdir(cache)
            self.generators_ = {}
            self.generators_['train'] = self._writeHdf5(cache, self.data_['train'], 'train')
            self.generators_['test'] = self._writeHdf5(cache, self.data_['test'], 'test')

    def _sort_xy(self, data_df, nSample, normalize = True, task = 'regression'):
        """
        sets last column to be Y
        """
        if task == 'classification':
            df = data_df[self.indep_pars_]
            dep_par_name = 'dependent_{}'.format(self.dep_pars_[0])
            cols = df.columns
        else:
            df = data_df[self.indep_pars_ + self.dep_pars_]
            cols = df.columns[:-1].tolist() + ['dependent_{}'.format(df.columns[-1])]
            df.columns = cols

        if normalize:
            scaler = sklearn.preprocessing.StandardScaler() if not self.cache_ else dask_ml.preprocessing.StandardScaler()
            df = scaler.fit_transform(df)

        if not self.cache_:#should transfer back to pandas DataFrame
            df = pd.DataFrame(df, columns = cols, index = data_df.index)

        # randomize/permute the independent parameter if it's the same as the target parameter
        for p in self.dep_pars_:
            if p in self.indep_pars_:
                if self.cache_:
                    import dask.array as da
                    import dask.dataframe as dd
                    tmp = da.random.normal(size = nSample, chunks = tuple(df.map_partitions(len).compute()))
                    tmp = dd.from_dask_array(tmp, columns=['rand'], index = df.index)
                    df[p] = tmp['rand']
                else:
                    perm = np.random.permutation(df[p].values.flatten())
                    df[p] = perm

        if task == 'classification':
            if self.cache_:
                import dask.array as da
                import dask.dataframe as dd
                tmp = dd.from_array(data_df[self.dep_pars_].values.flatten().reshape(-1,1), columns=['y'], index = df.index)
                df[self.dep_pars_] = tmp['y']
            else:
                df.insert(df.shape[-1], dep_par_name, data_df[self.dep_pars_])

        return df


    def _split(self, df, test_size, random_state = None):
        data = {}
        if not self.cache_:
            split = sklearn.model_selection.train_test_split
            data['train'], data['test'] = split(df, random_state = random_state, test_size = test_size)
        else:
            data['train'], data['test'] = df.random_split([1-test_size, test_size], random_state = random_state)
        return data

    def sample(self, n, split = 'train'):
        """
        n should be an integer, a data with approximately n samples is returned
        We used frac argument since "n" is not supported in dask dataframe
        """
        frac = min(1, np.round(n/self.nSample[split],4))
        return self.data_[split].sample(frac = frac, replace = False)

    #def _writeHdf5(self, path, data, name):
    #    with h5py.File(path+'/data_{}.hdf5'.format(self.label_), 'a') as f:
    #        f.create_dataset(name = name, data = data, dtype = 'float32')

    #    generator = generate_from_hdf5(path+'/data_{}.hdf5'.format(self.label_), name)
    #    return generator

    def _writeHdf5(self, path, data, name):
        data.to_hdf(path+'/data_{}.hdf5'.format(self.label_), key = name)
        generator = generate_from_hdf5(path+'/data_{}.hdf5'.format(self.label_), name)
        return generator

    def _append_to_generators(self, data, prefix = 'sample'):
        id = len(self.generators_.keys()) - 2 #there are two generators by default ('train' and 'test')
        name = '{}_{}'.format(prefix,id)
        self.generators_[name] = self._writeHdf5(self.cache_, data, name)
        return name

    def batch(self, split, size, shuffle = True, sample_n = None):
        if not self.cache_:
            raise ValueError('batch creation is only supported when an hdf5 cache folder is specified.')

        if sample_n:
            sampled_date = self.sample(sample_n, split)
            split = self._append_to_generators(sampled_date)

        nPar = len(self.indep_pars_) + 1
        ds = tf.data.Dataset.from_generator(self.generators_[split], tf.float32, tf.TensorShape([nPar]))
        if shuffle:
            ds = ds.shuffle(buffer_size = 1000, reshuffle_each_iteration = True)

        return ds.batch(size, drop_remainder = False)

    def getDatadf(self, split = 'train'):
        """
        returns dask dataframe if self.cache_ is specified, otherwise returns pandas dataframe
        """
        if split not in ['train','test']:
            raise ValueError('specified {} split is not available.'.format(split))

        return self.data_[split]
