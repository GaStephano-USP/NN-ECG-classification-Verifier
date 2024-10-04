import numpy as np
from scipy import signal
from pandas.core.nanops import nanmean as pd_nanmean


def seasonal_decompose(x, model="additive", filt=None, freq=None, two_sided=True,
                       extrapolate_trend=0):
    x = np.asanyarray(x).squeeze()
    nobs = len(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("This function does not handle missing values")
    if model.startswith('m'):
        if np.any(x <= 0):
            raise ValueError("Multiplicative seasonality is not appropriate "
                             "for zero and negative values")

    if freq is None:
        raise ValueError("You must specify a freq")

    if filt is None:
        if freq % 2 == 0:  # split weights at ends
            filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
        else:
            filt = np.repeat(1./freq, freq)

    nsides = int(two_sided) + 1
    trend = convolution_filter(x, filt, nsides)

    if extrapolate_trend == 'freq':
        extrapolate_trend = freq - 1

    if extrapolate_trend > 0:
        trend = _extrapolate_trend(trend, extrapolate_trend + 1)

    if model.startswith('m'):
        detrended = x / trend
    else:
        detrended = x - trend

    period_averages = seasonal_mean(detrended, freq)

    if model.startswith('m'):
        period_averages /= np.mean(period_averages, axis=0)
    else:
        period_averages -= np.mean(period_averages, axis=0)

    seasonal = np.tile(period_averages.T, nobs // freq + 1).T[:nobs]

    if model.startswith('m'):
        resid = x / seasonal / trend
    else:
        resid = detrended - seasonal

    return x, seasonal, trend, resid


def convolution_filter(x, filt, nsides=2):
    # for nsides shift the index instead of using 0 for 0 lag this
    # allows correct handling of NaNs
    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt)/2.) - 1) or None
        trim_tail = int(np.ceil(len(filt)/2.) - len(filt) % 2) or None
    else:  # pragma : no cover
        raise ValueError("nsides must be 1 or 2")
    x = np.asarray(x)
    filt = np.asarray(filt)
    if x.ndim > 1 and filt.ndim == 1:
        filt = filt[:, None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')

    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode='same')
    elif filt.ndim == 2:
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        if nsides == 2:
            for i in range(nvar):
                # could also use np.convolve, but easier for swiching to fft
                result[:, i] = signal.convolve(x[:, i], filt[:, i],
                                               mode='same')
        elif nsides == 1:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], np.r_[0, filt[:, i]],
                                               mode='same')
    return result


def _extrapolate_trend(trend, npoints):
    front = next(i for i, vals in enumerate(trend)
                 if not np.any(np.isnan(vals)))
    back = trend.shape[0] - 1 - next(i for i, vals in enumerate(trend[::-1])
                                     if not np.any(np.isnan(vals)))
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)

    k, n = np.linalg.lstsq(
        np.c_[np.arange(front, front_last), np.ones(front_last - front)],
        trend[front:front_last], rcond=-1)[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra

    k, n = np.linalg.lstsq(
        np.c_[np.arange(back_first, back), np.ones(back - back_first)],
        trend[back_first:back], rcond=-1)[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1:] = extra

    return trend


def seasonal_mean(x, freq):
    return np.array([pd_nanmean(x[i::freq], axis=0) for i in range(freq)])
