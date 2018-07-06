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


def num_grad(fn, X, h=1e-8, verbose=True):
    """
    Calculate finite differences gradient of function fn evaluated at numpy
    array X. Not calculating central diff to improve speed and because some
    authors disagree about benefits.

    Most of the code here is really to deal with weird sizes of inputs or
    outputs. If scalar input and multi-dim output or vice versa, we return
    gradient in the shape of the multi-dim input or output. However, if both
    are multi-dimensional then we return as n_output vs n_input matrix
    where both input/outputs have been vectorised if necessary.
    """
    assert callable(fn), "fn is not a function"
    assert isinstance(X, (np.ndarray, int, float)), "X should be a numpy array"
    X = np.asarray(X)
    assert isinstance(h, float), "h should be of type float"

    shp = X.shape
    resize_x = X.ndim > 1
    rm_xdim = X.ndim == 0
    n = X.size

    f_x = fn(X)
    if isinstance(f_x, (int, float)):
        im_f_shp = 0
        resize_y = False
    else:
        im_f_shp = f_x.shape
        resize_y = not (f_x.ndim == 0 or (f_x.ndim <= 2 and np.any(f_x.shape == 1)))
        assert f_x.ndim <= 2, "image of fn is tensor. Not supported."

    m = np.prod(np.maximum(im_f_shp, 1)).astype(int)

    X = X.ravel()
    g = np.ones((m, n))
    for ii in range(n):
        Xplus = X.copy().astype('float64')
        Xplus[ii] += h
        Xplus = Xplus.reshape(shp)
        grad = (fn(Xplus) - f_x) / h
        g[:, ii] = grad.ravel()

    if verbose and (resize_x and resize_y):
        warnings.warn("Returning gradient as matrix size n(fn output) x n(variables)")

    if rm_xdim and g.shape[1] == 1:
        g = g.ravel()
    elif resize_x and not np.any(im_f_shp > 1):
        g = g.reshape(shp)
    elif resize_y and not np.any(np.array(shp) > 1):
        g = g.reshape(im_f_shp)

    return g