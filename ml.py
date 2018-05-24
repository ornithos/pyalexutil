import numpy as np
import torch

def split_indices_kfold(k, ixs):
    if isinstance(ixs, int):
        ixs = np.arange(ixs)
    else:
        assert isinstance(ixs, np.ndarray), "indices must be either an integer maximum, or numpy ndarray."
        assert ixs.ndim == 1, "indices (numpy array) must be one dimensional."

    N  = ixs.size
    ok = np.sort(np.array([[x, N // x] for x in range(1, np.floor(np.sqrt(N)).astype(int))
                      if N % x == 0]).flatten())
    assert N % k == 0, "k does not divide N. Choose k in {:s}.".format(str(ok))
    m = N // k

    return [[ixs[np.concatenate((np.arange(0,ii*m), np.arange((ii+1)*m, N)))],
           ixs[np.arange(ii*m,(ii+1)*m)]] for ii in range(k)]


def istensor(x):
    """
    (torch) Helper function to evaluate if x is any form of tensor
    """
    return isinstance(x, torch.tensor._TensorBase)


def mse(x, y, axis=None, sqrt=False):

    assert axis is None or isinstance(axis, int), "axis should be an int"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x is {:s}, y is {:s}. Not today thanks.".format(str(x.shape), str(y.shape))

    sqdiff = (x - y)**2
    if axis is None:
        out =  np.nanmean(sqdiff)
    else:
        out = np.nanmean(sqdiff, axis=axis)

    if sqrt:
        return np.sqrt(out)
    else:
        return out


def mse_decomp(x, y, axis=None):

    assert axis is None or isinstance(axis, int), "axis should be an int"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x is {:s}, y is {:s}. Not today thanks.".format(str(x.shape), str(y.shape))

    def _mse_decomp(u):
        mudeltasq = np.nanmean(u)**2
        return np.array([np.nanvar(u), mudeltasq])

    if axis is None:
        out =  _mse_decomp(x.reshape(-1) - y.reshape(-1))
    else:
        out = np.apply_along_axis(_mse_decomp, axis=axis, arr=x-y)

    return out