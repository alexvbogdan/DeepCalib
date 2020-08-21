__author__ = 'Sergey Matyunin'

import numpy as np


def interp2linear(z, xi, yi, extrapval=np.nan):

    """
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB

    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required
    @param yi: matrix of y coordinates where interpolation is required
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """

    x = xi.copy()
    y = yi.copy()
    nrows, ncols, nchannels = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")


    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1))
    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    z_ravel0 = z[:,:,0].ravel()
    z_ravel1 = z[:,:,1].ravel()
    z_ravel2 = z[:,:,2].ravel()
    f0 = (z_ravel0[ndx] * one_minus_t + z_ravel0[ndx + ncols] * y ) * (1 - x) + (
            z_ravel0[ndx + 1] * one_minus_t + z_ravel0[ndx + ncols + 1] * y) * x
    f1 = (z_ravel1[ndx] * one_minus_t + z_ravel1[ndx + ncols] * y) * (1 - x) + (
            z_ravel1[ndx + 1] * one_minus_t + z_ravel1[ndx + ncols + 1] * y) * x
    f2 = (z_ravel2[ndx] * one_minus_t + z_ravel2[ndx + ncols] * y) * (1 - x) + (
            z_ravel2[ndx + 1] * one_minus_t + z_ravel2[ndx + ncols + 1] * y) * x
    f = np.stack([f0,f1,f2], axis=-1)
    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f
