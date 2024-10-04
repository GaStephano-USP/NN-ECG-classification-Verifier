import matplotlib.pyplot as plt
import numpy as np
from arch.bootstrap import MovingBlockBootstrap as MBB
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.nonparametric.smoothers_lowess import lowess
from oversampling.seasonal_decompose import *


def stl_bagging(timeseries, num_samples=10):
    timeseries, removed_padding = preprocess(timeseries)
    bcox, boxcox_lambda = apply_boxcox(timeseries, boxcox_lambda=None)
    components = get_components(bcox)
    plot_components(bcox, components)
    additional_signals = generate_additional_samples(
        timeseries, components, num_samples, boxcox_lambda=boxcox_lambda)
    return add_padding_back(additional_signals, removed_padding)


def preprocess(timeseries):
    timeseries = timeseries.astype(np.float64)
    timeseries, removed_padding = remove_zero_padding(timeseries)
    return remove_zeros(timeseries), removed_padding


def remove_zero_padding(timeseries):
    padding_begin = find_begining_of_padding(timeseries)
    timeseries_no_padding = timeseries[:padding_begin]
    return timeseries_no_padding, len(timeseries)-padding_begin


def find_begining_of_padding(timeseries):
    index = len(timeseries)-1
    while index > -1:
        if timeseries[index] != 0:
            return index+1
        index -= 1
    return 0


def remove_zeros(timeseries):
    eta = 1e-10
    return timeseries + eta


def add_padding_back(signals, removed_padding):
    padded = np.zeros((signals.shape[0], signals.shape[1]+removed_padding))
    for i in range(signals.shape[0]):
        padded[i][:signals[i].size] = signals[i]
    return padded


def apply_boxcox(timeseries, boxcox_lambda=0.9):
    return boxcox(timeseries.flatten(), lmbda=boxcox_lambda)


def get_components(bcox):
    is_seasonal = True
    if is_seasonal:
        _, seasonal, trend, remainder = seasonal_decompose(
            bcox, model='additive', freq=82, two_sided=False)
    else:
        x = np.arange(len(bcox))
        trend = lowess(bcox, x, frac=0.6666666666666666, it=10,
                       is_sorted=True, return_sorted=False)
        remainder = bcox - trend
        seasonal = np.zeros(trend.shape)
    return trend, seasonal, remainder


def plot_components(timeseries, components):
    trend, seasonal, remainder = components

    plt.figure(0, figsize=(10, 6))
    ax = plt.subplot(411)
    ax.plot(timeseries)
    plt.ylabel('timeseries')

    ax = plt.subplot(412)
    ax.plot(seasonal)
    plt.ylabel('seasonal')

    ax = plt.subplot(413)
    ax.plot(trend)
    plt.ylabel('trend')

    ax = plt.subplot(414)
    ax.plot(remainder)
    plt.ylabel('remainder')


def generate_additional_samples(timeseries, components, num_samples, boxcox_lambda=0.00001):
    trend, seasonal, remainder = components
    samples = []
    use_MBB = False
    if use_MBB:
        reminders = get_MBB_reminders(num_samples, remainder)
    else:
        reminders = np.random.randn(
            num_samples, seasonal.size) / 50 + remainder
    for i in range(num_samples):
        bc = trend + seasonal + reminders[i]
        samples.append(inv_boxcox(
            bc, boxcox_lambda).reshape(1, len(timeseries)))
    return np.concatenate(samples, axis=0)


def get_MBB_reminders(num_samples, remainder):
    reminders = np.zeros((num_samples, remainder.size))
    bs = MBB(3, remainder)
    for i, data in enumerate(bs.bootstrap(num_samples)):
        reminders[i] = data[0][0]
    return reminders
