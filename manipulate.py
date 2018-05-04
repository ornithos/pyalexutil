import numpy as np
import warnings

def vec(x):
    """
    Make column vector from multidimensional array.
    """
    return x.reshape(-1,1)


def one_hot(x, max_value=None):
    """
    :param x: x (array, length n)
    :param max_value: (optional, length p of output vectors)
    :return: (matrix n * p)
     Converts an array of integers into the 1-of-k coding suitable
     for comparison with argmax-like functions.
     See also one-liner from mattjj -- TODO: comparison of timing...
     one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    dims = x.shape
    narg = np.argmax(dims)
    n    = dims[narg]
    assert np.prod(dims) == np.max(dims), 'x must be one-dimensional array'
    assert x.dtype.kind == 'i', 'x must be of integer type'

    # Determine length of output vectors
    amxval = np.argmax(x)
    maxval = x[amxval]+1
    if max_value is not None:
        if max_value < maxval:
            warnings.warn('one_hot: max_value given is less than the maxval of array', RuntimeWarning)
    else:
        max_value = maxval[0]

    out    = np.zeros((n, max_value), dtype=int)

    if narg > 0:
        x      = x.reshape(-1,1)

    out[np.arange(n), x.T] = 1
    return out


def make_array_from_list(x):
    """
    Turn a list of numpy arrays (1D list of 1D numpy arrays) into 2D numpy array.
    The advantage of this function is that the numpy arrays do not need to be of
    the same length: this routine pads the arrays of shorter length with NaNs.
    :param x: a list of 1D numpy arrays
    :return: a 2D numpy array
    """
    assert isinstance(x, list), "x must be a list of numpy arrays (not a list!)"
    assert isinstance(x[0], np.ndarray), "x must be a list of numpy arrays"
    if x[0].ndim > 1:
        raise NotImplementedError("Only implemented for 1D arrays at present")

    max_len = max([z.size for z in x]) # feels a little inefficient, but naively, reduce doesn't help
    x = [np.pad(y, (0, max_len - y.size), 'constant', constant_values=np.nan) for y in x]
    return np.vstack(x)


def run_length_encoding(x):
    """
    For 1D array x, turn [0,0,0,0,1,1,1,1,0,1,1,1] into [0, 3, 7, 8], [3, 4, 1, 4], [0, 1, 0, 1]
    This will work with non boolean arrays but the final return element will not be meaningful.
    :param x:
    :return: (indices of changes, length of runs, new number (assuming boolean).
    """
    x = np.asarray(x)
    assert x.ndim == 1, "run_length_encoding currently only supports 1D arrays"

    changes = x[:-1] != x[1:]
    changes_ix = np.where(changes)[0]
    changes_from = np.concatenate(([int(not x[0])], x[changes_ix]))
    changes_ix = np.concatenate(([0], changes_ix))
    changes_to = np.logical_not(changes_from).astype(int)
    lengths = np.diff(np.concatenate((changes_ix, [x.size])))
    return changes_ix, lengths, changes_to