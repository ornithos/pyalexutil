import numpy as np
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
