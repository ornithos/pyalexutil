import numpy as np
import matplotlib.pyplot as plt

def subplot_gridsize(num):
    """ stolen from Matt Johnson - much better than my version
    https://github.com/mattjj/pymattutil/plot.py"""
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))

def vary_color(n):
    # Not suggesting this is a good colour palette
    # but it is merely consistent with my previous
    # MATLAB version
    cols           = np.array(
                     [[77,  77,  77 ],
                      [93,  165, 218],
                      [250, 164, 58 ],
                      [96,  189, 104],
                      [241, 124, 176],
                      [178, 145, 47 ],
                      [178, 118, 178],
                      [222, 207, 63 ],
                      [241, 88,  84 ]])

    assert isinstance(n, int), 'n must be an integer'
    assert 0 <= n < 9, 'vary_color only supports a 9 colour palette'

    return cols[n,:]/255;

def colortint(x, amt):
    # xtint = colortint(x, amt)
    # Creates a lighter tint of color x (where x is a n * 3 matrix of rgb
    # values).
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    if x.ndim > 1:
        assert x.shape[1]==3, "x must be an n*3 numpy array"
    else:
        assert x.size == 3, "x must be an n*3 2D numpy array, or size 3 array"

    n = x.shape[0]
    assert np.all(0 <= x) and np.all(x <= 1), "x must contain values only in [0,1]"
    assert isinstance(amt, int) or isinstance(amt, float), "amt must be numeric"
    assert amt >= 0 and amt <= 2, "amt must be scalar in [0,2]"

    if amt <= 1:
        xtint = 1 - x;
        xtint = 1 - amt*xtint;
    else:
        non1  = x < 1;
        xtint = x;
        xtint[non1] = (2-amt)*x[non1];
    return xtint


def abline(slope, intercept, color='red', ax=None):
    """Plot a line from slope and intercept - improved from https://stackoverflow.com/a/43811762"""
    if ax is None:
        ax = plt.gca()
    xrng = ax.get_xlim()
    yrng = ax.get_ylim()
    yrng2 = ((yrng[1]-intercept)/slope, (yrng[0]-intercept)/slope)
    x_vals = [max(xrng[0], min(yrng2)), min(max(yrng2), xrng[1])]
    x_vals = np.array(x_vals)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color=color)