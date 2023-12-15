#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from tqdm import tqdm

tqdm.pandas()
import math

#%%
def Add(data1, data2):
    return data1 + data2


def Sub(data1, data2):
    return data1 - data2


def Mul(data1, data2):
    return data1.multiply(data2)


def Div(data1, data2):
    return data1.div(data2.replace(0, 0.001))


def inv(data):
    d = 1 / data
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)
    return d


def sqrt(data):
    data = data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.apply(lambda x: x.apply(abs).apply(math.sqrt))


def sin(data):
    data = data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.apply(lambda x: x.apply(math.sin))


def cos(data):
    data = data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.apply(lambda x: x.apply(math.cos))


def abslog(data):
    data = data.apply(lambda x: x.apply(abs)).replace(0, np.nan)
    d = data.apply(lambda x: x.apply(math.log))
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d


def quantile_3_60(data):
    return data.rolling(60).quantile(0.75)


def quantile_1_60(data):
    return data.rolling(60).quantile(0.25)


def quantile_3_120(data):
    return data.rolling(120).quantile(0.75)


def quantile_1_120(data):
    return data.rolling(120).quantile(0.25)


def max_(data1, data2):
    """max of two number"""
    return (data1 + data2) / 2 + ((data1 - data2) / 2).apply(lambda x: x.apply(abs))


def min_(data1, data2):
    """min of two number"""
    return (data1 + data2) / 2 - ((data1 - data2) / 2).apply(lambda x: x.apply(abs))


def mean5(data):
    return data.rolling(5).mean()


def mean10(data):
    return data.rolling(10).mean()


def mean20(data):
    return data.rolling(20).mean()


def mean60(data):
    return data.rolling(60).mean()


def mean120(data):
    return data.rolling(120).mean()


def std10(data):
    return data.rolling(10).std()


def std20(data):
    return data.rolling(20).std()


def std60(data):
    return data.rolling(60).std()


def std120(data):
    return data.rolling(120).std()


def max10(data):
    return data.rolling(10).max()


def max20(data):
    return data.rolling(20).max()


def max60(data):
    return data.rolling(60).max()


def max120(data):
    return data.rolling(120).max()


def min10(data):
    return data.rolling(10).min()


def min20(data):
    return data.rolling(20).min()


def min60(data):
    return data.rolling(60).min()


def min120(data):
    return data.rolling(120).min()


def shift10(data):
    return data.shift(10)


def shift20(data):
    return data.shift(20)


def shift60(data):
    return data.shift(60)


def shift120(data):
    return data.shift(120)


def square(data):
    return data.multiply(data)


def rank10(data):
    return data.rolling(10).apply(lambda x: x.rank()[-1])


def rank20(data):
    return data.rolling(20).apply(lambda x: x.rank()[-1])


def rank60(data):
    return data.rolling(60).apply(lambda x: x.rank()[-1])
