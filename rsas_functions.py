# -*- coding: utf-8 -*-
"""
Python translation of Cython `_rsas_functions.pyx`, preserving logic 1:1.
"""
import numpy as np
import scipy.stats
from scipy.special import gamma as gamma_function, gammainc, gammaincinv, erfc
from scipy.interpolate import interp1d

# for debugging
debug = True

def _verbose(statement):
    """Print debugging messages if debug==True"""
    if debug:
        print(statement)


def create_function(rSAS_type, params):
    """Initialize an rSAS function"""
    function_dict = {
        'gamma': _gamma_rSAS,
        'uniform': _uniform_rSAS,
        'kumaraswami': _kumaraswami_rSAS,
        'invgauss': _invgauss_rSAS,
        'lookuptable': _lookup_rSAS
    }
    if rSAS_type in function_dict:
        return function_dict[rSAS_type](params)
    elif hasattr(scipy.stats, rSAS_type):
        return _stats_rSAS(rSAS_type, params)
    else:
        raise ValueError('No such rSAS function type')


def make_lookup(rSAS_fun, P_list=None, NP=101):
    if P_list is not None:
        if type(P_list) is not np.ndarray:
            P_list = np.array(P_list)
        if P_list.ndim != 1:
            raise TypeError('P_list must be a 1-D array')
        if P_list[-1] != 1:
            raise TypeError('P_list[-1] must be 1')
        if P_list[0] != 0:
            raise TypeError('P_list[0] must be 0')
        if not all(P_list[i] <= P_list[i+1] for i in range(len(P_list)-1)):
            raise TypeError('P_list must be sorted')
    else:
        P_list = np.linspace(0,1,NP)
    NP = len(P_list)
    N = len(rSAS_fun.ST_min)
    fun_methods = [m for m in dir(rSAS_fun) if callable(getattr(rSAS_fun, m))]
    if not ('cdf_all' in fun_methods and 'cdf_i' in fun_methods):
        raise TypeError('Each rSAS function must have methods cdf_all and cdf_i')
    rSAS_lookup = np.zeros((len(P_list), N))
    for i in range(N):
        rSAS_lookup[:, i] = rSAS_fun.invcdf_i(P_list, i)
        rSAS_lookup[0, i] = rSAS_fun.ST_min[i]
    return P_list, rSAS_lookup


def convert_to_lookup(rSAS_fun, **kwargs):
    return create_function('lookuptable', make_lookup(rSAS_fun, **kwargs))


class rSASFunctionClass:
    def __init__(self, params):
        raise NotImplementedError('__init__ not implemented in derived rSASFunctionClass')
    def cdf_all(self, ST):
        raise NotImplementedError('cdf_all not implemented in derived rSASFunctionClass')
    def cdf_i(self, ST, i):
        raise NotImplementedError('cdf_i not implemented in derived rSASFunctionClass')


class _kumaraswami_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.a = params[:,2]
        self.b = params[:,3]
    def cdf_all(self, ST):
        return np.where(self.ST_max>=self.ST_min,
                        np.where(ST>self.ST_min,
                                 np.where(ST<self.ST_max,
                                          1 - (1 - ((ST-self.ST_min)/(self.ST_max-self.ST_min))**self.a)**self.b,
                                          1.),
                                 0.),
                        np.where(ST>self.ST_min,
                                 1 - (1 - ((ST-self.ST_min)/(self.ST_max-self.ST_min))**self.a)**self.b,
                                 0.))

    # def cdf_i(self, ST, i):
    #     return np.where(self.ST_max[i]>=self.ST_min[i],
    #                     np.where(ST>self.ST_min[i],
    #                              np.where(ST<self.ST_max[i],
    #                                       1 - (1 - ((ST-self.ST_min[i])/(self.ST_max[i]-self.ST_min[i]))**self.a[i])**self.b[i],
    #                                       1.),
    #                              0.),
    #                     np.where(ST>self.ST_min[i],
    #                              1 - (1 - ((ST-self.ST_min[i])/(self.ST_max[i]-self.ST_min[i]))**self.a[i])**self.b[i],
    #                              0.))
    def cdf_i(self, ST, i): #和cdf_all的区别是 ST_min与ST_max随着时间i的取值可以不同
        if self.ST_max[i] <= self.ST_min[i]:
            return np.where(ST > self.ST_min[i], 1., 0.)

        # 确保分子非负
        numerator = np.maximum(ST - self.ST_min[i], 0)
        denominator = self.ST_max[i] - self.ST_min[i]

        # 限制比值在[0, 1]范围内
        ratio = np.minimum(numerator / denominator, 1)

        # 计算CDF
        cdf = 1 - (1 - ratio ** self.a[i]) ** self.b[i]

        # 处理边界
        return np.where(ST <= self.ST_min[i], 0.,
                        np.where(ST >= self.ST_max[i], 1., cdf))


class _uniform_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.lam = 1.0/(self.ST_max-self.ST_min)
    def cdf_all(self, ST):
        return np.where(ST<self.ST_max,
                        np.where(ST>self.ST_min,
                                 self.lam*(ST-self.ST_min),
                                 0.),
                        1.)
    def cdf_i(self, ST, i):
        return np.where(ST<self.ST_max[i],
                        np.where(ST>self.ST_min[i],
                                 self.lam[i]*(ST-self.ST_min[i]),
                                 0.),
                        1.)


class _invgauss_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.loc = params[:,0]
        self.scale = params[:,1]
        self.mu = params[:,2:]
    def cdf_i(self, ST, i):
        x = (ST-self.loc[i]) / self.scale[i]
        return (erfc((-x+self.mu[i])/(np.sqrt(2*x)*self.mu[i]))
                + np.exp(2/self.mu[i])*erfc((x+self.mu[i])/(np.sqrt(2*x)*self.mu[i]))) / 2.


class _stats_rSAS(rSASFunctionClass):
    def __init__(self, rSAS_type, params):
        params = params.copy()
        self.dist_class = getattr(scipy.stats, rSAS_type)
        self.loc = params[:,0]
        self.scale = params[:,1]
        N = params.shape[0]
        if params.shape[1] > 2:
            self.shape = params[:,2:]
            self.dist = [self.dist_class(*self.shape[i], loc=self.loc[i], scale=self.scale[i])
                         for i in range(N)]
        else:
            self.dist = [self.dist_class(loc=self.loc[i], scale=self.scale[i])
                         for i in range(N)]
    def cdf_i(self, ST, i):
        return [self.dist[i].cdf(STi) for STi in ST]


class _gamma_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.scale = params[:,2] #尺度参数 越大 约接近均匀分布
        self.a = params[:,3] #形状参数 a越大
        self.lam = 1.0/self.scale
        self.lam_on_gam = self.lam**self.a / gamma_function(self.a)
        self.rescale = np.where(np.isfinite(self.ST_max),
                                1/(gammainc(self.a, self.lam*(self.ST_max-self.ST_min))),
                                1.)
    def cdf_all(self, ST):
        return np.where(ST>self.ST_min,
                        np.where(ST<self.ST_max,
                                 gammainc(self.a, self.lam*(ST-self.ST_min))*self.rescale,
                                 1.),
                        0.)
    def cdf_i(self, ST, i):
        return np.where(ST>self.ST_min[i],
                        np.where(ST<self.ST_max[i],
                                 gammainc(self.a[i], self.lam[i]*(ST-self.ST_min[i]))*self.rescale[i],
                                 1.),
                        0.)
    def invcdf_i(self, P, i):
        return (np.where(P>0,
                         np.where(P<1,
                                  gammaincinv(self.a[i], P/self.rescale[i]),
                                  np.inf),
                         np.nan)
                / self.lam[i] + self.ST_min[i])


class _lookup_rSAS(rSASFunctionClass):
    def __init__(self, params):
        self.P_list = params[0].copy()
        self.rSAS_lookup = params[1].copy()
        self.ST_min = self.rSAS_lookup[0, :]
        self.ST_max = self.rSAS_lookup[-1, :]
        if not (self.P_list[0]==0 and self.P_list[-1]==1):
            raise ValueError('The first and last value of S_T must correspond with probability 0 and 1 respectively')
        self.interpfuns = []
        for i in range(len(self.ST_min)):
            self.interpfuns.append(
                interp1d(self.rSAS_lookup[:,i], self.P_list,
                         kind='linear', copy=False,
                         bounds_error=False, assume_sorted=True)
            )
    def cdf_all(self, ST):
        return None
    def cdf_i(self, ST, i):
        return np.where(ST<self.ST_max[i],
                        np.where(ST>self.ST_min[i], self.interpfuns[i](ST), 0.),
                        1.)
