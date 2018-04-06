import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import seaborn as sns
import colorsys


def get_seq_cmap_by_snscolor(cix):
    color = sns.palettes.color_palette()[cix]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.utils.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0.4, 12)]
    cmap = sns.palettes.blend_palette(colors, as_cmap=True)
    return cmap


def rgb_reduce_hls(x, h=None, s=None, l=None):
    assert isinstance(x, (tuple, list))
    assert len(x) == 3, "x must have 3 elements (r,g,b)"

    rgb = mplcol.colorConverter.to_rgb(x)

    # Convert to hls
    x_h, x_l, x_s = colorsys.rgb_to_hls(*rgb)
    x = dict(h=x_h, l=x_l, s=x_s)

    for nm, a in zip('hls', [h,l,s]):
        if a is not None:
            assert isinstance(a, (float,int)) and 0 <= a <= 10, "{:s} should be a float between 0 and 10".format(nm)
            x[nm] = x[nm] * a
            if x[nm] > 1:
                warnings.warn("{:s} level clipped at 1.".format(nm))
                x[nm] = 1

    return colorsys.hls_to_rgb(x['h'], x['l'], x['s'])


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


class non_uniform_subplot(object):
    def __init__(self, ratio_y, ratio_x):
        self.gs = gridspec.GridSpec(len(ratio_y), len(ratio_x), width_ratios=ratio_x, height_ratios=ratio_y)
        self.ny = len(ratio_y)
        self.nx = len(ratio_x)
        self.ratio_y = ratio_y
        self.ratio_x = ratio_x

    def subplot(self, i, j=None):
        assert i > 0, 'subplot syntax is 1-based'
        if j is not None:
            assert j > 0, 'subplot syntax is 1-based'
            i = (i - 1) * self.nx + j

        ax = plt.subplot(self.gs[i - 1])
        return ax


def create_grid_line_plot(ax, y, x=None, max_x=None, title=None, ylab=None, xlab=None,
                          hline=None, ylimZero=True):
    assert x is not None or max_x is not None, "either x or max_x must be given."
    if x is not None:
        if max_x is not None:
            x_mask = np.array([a <= max_x for a in x ])
            x = x[x_mask]
        else:
            max_x = len(x)
            x_mask = np.arange(max_x)
    else:
        x = np.arange(max_x)
        x_mask = np.arange(max_x)

    f = plt.plot(x, np.atleast_2d(y)[:,x_mask].T)
    if ylimZero:
        ax.set_ylim(ymin=0)
    if title:
        plt.title(title)
    if ylab:
        plt.ylabel(ylab)
    if xlab:
        plt.xlabel(xlab)
    if hline:
        ax.axhline(hline, dashes=[4,2], color='red', alpha=0.6)
    ax.set_xticks(np.arange(0,max_x,3))
    ax.set_xticks(np.arange(max_x), minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    return f


def kdeplot2d(x, y, ax, bw="scott", gridsize=100, cut=3, n_levels=10, col_ix=None,
            fill_lowest=False, sgl_col=False, cbar=False, x_label=None, y_label=None,
            xclip=None, **kwargs):
    # butchered from Seaborn's distributions._bivariate_kdeplot

    clip = [(np.min(x), np.max(x)), (-np.inf, np.inf)]
    # override some settings if in single color mode.
    if sgl_col:
        if fill_lowest:
            fill_lowest=False
            warnings.warn("sgl_col overriding fill_lowest=True --> False")
        alpha /= 4


    if xclip is not None:
        xclip = [xclip] if isinstance(xclip, (int, float)) else xclip
        assert isinstance(xclip, (tuple, list)), "clip must be a list"
        tmpxclip = list(clip[0])
        tmpxclip[:len(xclip)] = np.array(xclip)
        clip[0] = tuple(tmpxclip)

    xx, yy, z = sns.distributions._statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    if 'cmap' not in kwargs:
        col_ix = 0 if col_ix is None else col_ix
        kwargs['cmap'] = get_seq_cmap_by_snscolor(col_ix)
    elif col_ix is not None:
        warnings.warn('supplied col_ix will be ignored since cmap specified.')

    cset = ax.contourf(xx, yy, z, n_levels, **kwargs)

    if sgl_col:
        alpha = 1 if 'alpha' not in kwargs else kwargs['alpha']
        for i, csetcoll in enumerate(cset.collections):
            if i > 18:
                csetcoll.set_alpha(0)
            else:
                # csetcoll.set_edgecolor(None)
                csetcoll.set_facecolor(kwargs['cmap'](256))
                csetcoll.set_linewidth(1e-2)
                csetcoll.set_alpha(alpha)

    if not fill_lowest:
        cset.collections[0].set_alpha(0)

    kwargs["n_levels"] = n_levels

    if cbar:
        ax.figure.colorbar(cset)

    # Label the axes
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax