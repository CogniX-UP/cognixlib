from mne_features import univariate, bivariate
import numpy as np

class Statistics:

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, data):
        self.func = None
        self.params = None
        self.data = data
    
    def calculate_stat(self):
        return self.func(**self.params)


class ComputeMean(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_mean
        self.params = {'data':data}
    
class ComputeVar(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_variance
        self.params = {'data':data}

class ComputeStd(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_std
        self.params = {'data':data}

class ComputePTPAmp(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_ptp_amp
        self.params = {'data':data}

class ComputeSkew(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_skewness
        self.params = {'data':data}

class ComputeKurtosis(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_kurtosis
        self.params = {'data':data}

class ComputeRMS(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_rms
        self.params = {'data':data}

class ComputeHjorthMobility(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_hjorth_mobility
        self.params = {'data':data}

class ComputeHjorthComplexity(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_hjorth_complexity
        self.params = {'data':data}

class Compute75Quantile(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_quantile
        self.params = {'data':data,'q':0.75}

class Compute25Quantile(Statistics):

    def __init__(self, data):
        self.func = univariate.compute_quantile
        self.params = {'data':data,'q':0.25}


