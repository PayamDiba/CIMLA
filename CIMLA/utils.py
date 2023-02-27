import os
import h5py

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class generate_from_hdf5:
    def __init__(self, file, name):
        self.file_ = file
        self.name_ = name

    def __call__(self):
        with h5py.File(self.file_, 'r') as hf:
            for d in hf[self.name_]['table']:#d[0] is index name and d[1] is data
                yield d[1]

class runningMean(object):
    def __init__(self):
        self.n = 0
        self.mean = 0

    def append(self, newMean, size):
        self.mean = (self.mean * self.n + newMean * size)/(self.n + size)
        self.n += size
