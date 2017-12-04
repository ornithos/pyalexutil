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
