import numpy as np
from scipy import stats
import sys

def max(x):
    """
    max, argmax = max(x)
    Return both max and argmax simultaneously
    """
    argmax = np.argmax(x)
    a = sys.isunix()
    return x[argmax], argmax


def log1pexp(x):
    """
    returns log(1 + exp(x)) in a more numerically stable way.
    http://sachinashanbhag.blogspot.co.uk/2014/05/numerically-approximation-of-log-1-expy.html
    Changepoints chosen empirically.
    :param x: ndarray
    :return: ndarray
    """
    linear_part = x > 35
    exp_part    = x < -10
    std_part    = np.invert(np.logical_or(linear_part, exp_part))

    ex  = np.exp(np.minimum(x, 35))
    std = np.log(1 + ex)
    out = std_part*std + exp_part*ex + linear_part*x
    return out


def nan_trim_mean(x, proportiontocut=0.1, axis=0, **kwargs):
    """
    Applies scipy's trim_mean function on a given axis of numpy array
    """
    assert isinstance(x, np.ndarray), "x is not numpy array"

    def _nan_trim_mean(x, proportiontocut, **kwargs):
        mask = np.isnan(x)
        x = x[~mask]
        return stats.trim_mean(x, proportiontocut, **kwargs)

    # This feels kind of inefficient, but not sure how easy it is to do better
    r = np.apply_along_axis(_nan_trim_mean, axis=axis, arr=x, proportiontocut=proportiontocut, **kwargs)

    return r


def softmax(x):
    m = np.max(x)
    x -= m
    y = np.exp(x)
    return y / sum(y)