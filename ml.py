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

