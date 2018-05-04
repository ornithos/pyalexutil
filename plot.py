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
import operator


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
    """ stolen from Matt Johnson - much cleaner than my version
    https://github.com/mattjj/pymattutil/plot.py"""
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))


def vary_color(n):
    # Not suggesting this is a good colour palette
    # but it is merely consistent with my previous
    # MATLAB version
    warnings.warn("This function will be deprecated in a future release - use matplotlib colormaps instead",
                  DeprecationWarning)

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
    warnings.warn("This function will be deprecated in a future release - use rgb_reduce_hls instead",
                  DeprecationWarning)
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


def ax_lim_one_side(ax, xy, start=None, end=None, type='constant'):
    if xy == 'x':
        lims = list(ax.get_xlim())
    else:
        lims = list(ax.get_ylim())

    if type in 'multiply*':
        f = operator.mul
    elif type == 'add+':
        f = operator.add
    elif type == 'constant':
        f = lambda x, y: y
    else:
        warnings.warn("Unexpected type (expecting 'constant', 'add', 'multiply'): interpreting as constant")
        f = lambda x, y: y

    if start is not None:
        lims[0] = f(lims[0], start)
    if end is not None:
        lims[1] = f(lims[1], end)

    if xy == 'x':
        ax.set_xlim(lims)
    else:
        ax.set_ylim(lims)


# convenience wrappers for clean code
def x_lim_one_side(ax, start=None, end=None, type='constant'):
    ax_lim_one_side(ax, 'x', start=start, end=end, type=type)


def y_lim_one_side(ax, start=None, end=None, type='constant'):
    ax_lim_one_side(ax, 'y', start=start, end=end, type=type)


def gaussian_2D_level_curve(mu, sigma, alpha=2, ncoods=100, plot=True):

    # (Ported from my matlab utils)
    assert isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray), "mu/sigma must be numpy arrays."
    assert mu.shape == (2, ), 'mu must be vector in R^2'
    assert sigma.shape == (2, 2), 'sigma must be 2x2 array'

    U, S, V = np.linalg.svd(sigma)

    sd = np.sqrt(S)
    coods = np.linspace(0, 2 * np.pi, ncoods)
    coods = np.vstack((sd[0] * np.cos(coods), sd[1] * np.sin(coods))) * alpha

    # project onto basis of ellipse
    coods = (V @ coods).T

    # add mean
    coods += mu

    if plot:
        plt.plot(*coods.T)

    return coods

class NonUniformSubplot(object):
    def __init__(self, ratio_y, ratio_x, f=None):
        if f is None:
            f = plt.figure()
        assert isinstance(f, mpl.figure.Figure), "supplied f is not a figure."
        self._gs = gridspec.GridSpec(len(ratio_y), len(ratio_x), width_ratios=ratio_x, height_ratios=ratio_y, figure=f)
        self._ny = len(ratio_y)
        self._nx = len(ratio_x)
        self._subplot_axes = [None]*len(ratio_y)*len(ratio_x)
        self.gs = self._gs
        self.nx = self._nx
        self.ny = self._ny
        self.restriction = None
        self.ratio_y = ratio_y
        self.ratio_x = ratio_x
        self.figure = f

    def subplot(self, i, j=None):
        assert i != 0, 'subplot syntax is 1-based'
        if j is not None:
            assert j != 0, 'subplot syntax is 1-based'
            i = i if i > 0 else self.ny + i + 1
            j = j if j > 0 else self.nx + i + 1
            i = (i - 1) * self.nx + j
        else:
            i = i if i > 0 else self.ny * self.nx + i + 1

        # get subplot axes, either saved from previous correction, or generated.
        # also error check bounds
        try:
            ax = self.subplot_axes(i-1)

            if ax is None:
                ax = self.figure.add_subplot(self.gs[i - 1])
                self._subplot_axes[self.get_absolute_ix(i-1)] = ax
        except IndexError as e:
            gt_all_subplot = i-1 >= self._gs._nrows * self._gs._ncols
            errstr = 'User specified index greater than exists in {:s}subplot ({:s}), i={:d}'.format(
                '(restricted) ' if self.is_restrict() and not gt_all_subplot else '',
                'x'.join([str(self.ny), str(self.nx)]), i)
            if gt_all_subplot:
                raise RuntimeError(errstr)
            else:
                raise RuntimeError(errstr + '\nTry unrestricting the subplot range (.unrestrict_subplot_range())')

        return ax

    def restrict_subplot_range(self, i_y=None, i_x=None):
        """
        make subplot method refer only to restricted part of grid: useful for pretending that
        a large grid is really a grid of subgrids, and potentially useful for client functions.
        Note at present that this is 0-based unlike the subplot command
        """
        assert i_y is None or isinstance(i_y, list), "i_y must be type list"
        assert i_x is None or isinstance(i_x, list), "i_x must be type list"
        i_y = list(range(self.ny)) if i_y is None else i_y
        i_x = list(range(self.nx)) if i_x is None else i_x
        ixs = np.concatenate([np.array(i_x) + y*self.nx for y in i_y])
        self.gs = [self._gs[i] for i in ixs]
        self.ny = len(i_y)
        self.nx = len(i_x)
        if self.ny == self._ny and self.nx == self._nx:
            self.restriction = None
        else:
            self.restriction = ixs

    def unrestrict_subplot_range(self):
        self.gs = self._gs
        self.ny = self._ny
        self.nx = self._nx
        self.restriction = None

    def is_restrict(self):
        return self.restriction is not None

    def get_absolute_ix(self, ix):
        if self.restriction is None:
            return ix
        else:
            return self.restriction[ix]

    def set_size_inches(self, w, h):
        self.figure.set_size_inches(w, h)

    def tight_layout(self, *args, **kwargs):
        self._gs.tight_layout(*args, **kwargs)

    def subplot_axes(self, ix):
        return self._subplot_axes[self.get_absolute_ix(ix)]


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
            fill_lowest=False, sgl_col=False, credint=0, cbar=False, x_label=None, y_label=None,
            xclip=None, **kwargs):
    # butchered from Seaborn's distributions._bivariate_kdeplot

    clip = [(np.min(x), np.max(x)), (-np.inf, np.inf)]
    alpha = 1 if 'alpha' not in kwargs else kwargs['alpha']
    # override some settings if in single color mode.
    if sgl_col:
        if fill_lowest:
            fill_lowest=False
            warnings.warn("sgl_col overriding fill_lowest=True --> False")
        alpha /= 4
    else:
        if credint > 0:
            alpha /= 4

    if xclip is not None:
        xclip = [xclip] if isinstance(xclip, (int, float)) else xclip
        assert isinstance(xclip, (tuple, list)), "clip must be a list"
        tmpxclip = list(clip[0])
        tmpxclip[:len(xclip)] = np.array(xclip)
        clip[0] = tuple(tmpxclip)

    xx, yy, z = sns.distributions._statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)

    #  ---------  credible interval ---------------------------------------
    if credint > 0:
        zsort = np.flipud(np.argsort(z, axis=0))
        zout = np.zeros_like(z)
        fillval = np.max(z)
        for i in range(zsort.shape[1]):
            column = zsort[:,i]
            vals = z[column, i]
            cumvals = np.cumsum(vals)
            cumvals /= cumvals[-1]
            is_1 = (cumvals <= credint)
            zout[zsort[is_1,i], i] = fillval #np.sum(vals)/np.sum(is_1)
        z = zout
        n_levels = 2
        # z[0, :] = 0
        # z[-1, :] = 0

    # -------- Plot the contours -----------------------------------
    if 'cmap' not in kwargs:
        col_ix = 0 if col_ix is None else col_ix
        kwargs['cmap'] = get_seq_cmap_by_snscolor(col_ix)
    elif col_ix is not None:
        warnings.warn('supplied col_ix will be ignored since cmap specified.')

    cset = ax.contourf(xx, yy, z, n_levels, **kwargs)

    # -------- switch up contour area colours ---------------------
    if sgl_col:
        for i, csetcoll in enumerate(cset.collections):
            if i > 18:
                # (AB): confused why this is here
                csetcoll.set_alpha(0)

            else:
                # csetcoll.set_edgecolor(None)
                csetcoll.set_facecolor(kwargs['cmap'](256))
                csetcoll.set_linewidth(1e-2)
                csetcoll.set_alpha(alpha)

    if sgl_col < 2 and not fill_lowest:
        cset.collections[0].set_alpha(0)

    if sgl_col == 0 and credint > 0:
        cset_collections[1].set_alpha(alpha)

    if sgl_col == 2 and not fill_lowest:
        cset.collections[0].set_linewidth(1)
        cset.collections[0].set_edgecolor(kwargs['cmap'](256))
        cset.collections[0].set_alpha(0.7)
        cset.collections[0].set_facecolor([1,1,1,0])

    kwargs["n_levels"] = n_levels

    if cbar:
        ax.figure.colorbar(cset)

    # Label the axes
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax


def nanboxplot(x, **kwargs):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]   # cannot use np.atleast_2d since adds adds dim at front not end
    no_nan_cols = [x[~np.isnan(x[:,i]), i] for i in range(x.shape[1])]
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        return ax.boxplot(no_nan_cols, **kwargs)
    else:
        return plt.boxplot(no_nan_cols, **kwargs)


def paired_boxplot(x, **kwargs):
    assert isinstance(x, list), "x must be a list"
    assert all(isinstance(xi, list) for xi in x), "x must be a list of lists"
    assert len(set(len(xi) for xi in x)) == 1, "all lists in x must be same length."

    def color_box(bp, color):
        # https://stackoverflow.com/a/35778857
        elements = ['boxes', 'caps', 'whiskers']
        for elem in elements:
            [plt.setp(bp[elem][idx], color=color) for idx in range(len(bp[elem]))]
        plt.setp(bp['medians'][0], color=rgb_reduce_hls(color[:3], s=0.7, l=0.9))
        return

    n_groups = len(x)
    n_x_axis = len(x[0])
    N = n_x_axis * (n_groups + 1)

    width = 0.6 if 'width' not in kwargs else kwargs.pop('width')
    _flierprops = None if 'flierprops' not in kwargs else kwargs.pop('flierprops')
    colormap = cm.tab10 if 'colormap' not in kwargs else kwargs.pop('colormap')

    col_ixs = [0] if n_groups == 1 else ([1,0] + list(range(2, n_groups)))[:n_groups]
    cols = [colormap(i) for i in  col_ixs]

    for n in range(n_groups):
        flierprops = dict(markeredgecolor=cols[n]) if _flierprops is None else _flierprops
        for i, pos in enumerate(range(n, N, n_groups+1)):
            bp = nanboxplot(x[n][i], positions=[pos+1], widths=width, flierprops=flierprops, **kwargs)
            color_box(bp, cols[n])

    ax = plt.gca() if 'ax' not in kwargs else kwargs['ax']

    ax.set_xlim(1-width, N + width-1)    # o.w. only see last plot
    ax.set_xticklabels([str(i) for i in range(n_x_axis)])
    ax.set_xticks([1 + (n_groups-1)/2 + (n_groups+1)*i for i in range(n_x_axis)])

    return ax