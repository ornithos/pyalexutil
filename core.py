import numpy as np
import builtins

def crange(start, stop, modulo):
    """
    crange(start, stop, modulo)
    Returns a generator for a *circular* range. Equivalent to range(start, stop) % modulo.
    :param start:
    :param stop:
    :param modulo:
    :return: generator
    """
    index = start
    while index != stop:
        yield index
        index = (index + 1) % modulo

# class MutableNamespace:
#     """
#     re-implementation of types.SimpleNamspace. Unfortunately
#     the (python 3) types.SimpleNamespace class is immutable,
#     which is not ideal for a namespace keeping track of
#     large objects. (Please forgive me if this is stupid,
#     I'm new to Python.)
#     Also somewhat similar to NamedTuple, but is extensible.
#     """
#
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#
#     def __repr__(self):
#         keys = sorted(self.__dict__)
#         items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
#         return "{}({})".format(type(self).__name__, ", ".join(items))
#
#     def __eq__(self, other):
#         return self.__dict__ == other.__dict__

def sort(x, reverse=False):
    assert isinstance(x, list), "utils sort is for lists only, otherwise use np.argsort"
    ix = sorted(range(len(x)), key=x.__getitem__, reverse=reverse)
    return [x[i] for i in ix], ix

def inverse_permutation(x):
    if isinstance(x, list):
        pass
    elif isinstance(x, np.ndarray):
        assert x.ndim == 1, "x must be 1-dimensional array"
        x = x.tolist()
    else:
        raise ValueError("x must be a list or np.ndarray")

    x = [[a,b] for a, b in enumerate(x)]
    x.sort(key=lambda q: q[1])
    return [a[0] for a in x]


def type(x, recurse_depth=np.Inf, prefix=""):
    def do_recurse(v):
        type(v, recurse_depth=recurse_depth - 1,
             prefix='   --- [0]: ' if len(prefix) == 0 else (' '*10) + prefix)
    if isinstance(x, list) or isinstance(x, tuple):
        print_str = prefix+"{0:s} [ length {1:d} ]".format(builtins.type(x).__name__, len(x))
        if recurse_depth > 0:
            print(print_str)
            do_recurse(x[0])
        else:
            first_el = builtins.type(x[0]).__name__
            print_str = print_str[:-2] + ", first element type {0:s} ]".format(first_el)
            print(print_str)
    elif isinstance(x, np.ndarray):
        shape = str(x.shape)
        print(prefix+"np.ndarray [ shape {:s}, dtype {:s} ]".format(shape, str(x.dtype)))
        if x.dtype.type is np.object_:
            do_recurse(x[0])
    elif isinstance(x, dict):
        keys = [x for x in x.keys() if not x[:2] == "__"]
        l    = len(keys)
        if l > 3:
            keys_str = "first 3/{:d} (non-internal) keys: ".format(l) + ", ".join(keys[:3]) + " ... "
        else:
            keys_str = "{:d} (non-internal) keys: ".format(l) + ", ".join(keys)

        print(prefix+"dict [ {:s} ]".format(keys_str))
    else:
        print(prefix+builtins.type(x).__name__)


class dictNamespace(object):
    """
    converts a dictionary into a namespace
    """
    def __init__(self, adict):
        self.__dict__.update(adict)
