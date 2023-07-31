#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:27:57 2022

@author: daddona
"""
import numpy as np
from scipy.ndimage import map_coordinates
from astropy.modeling import Fittable1DModel, Parameter


STD_PROGRESS_FMT = "\r{0:s}{1:s} {2:.2%}\r"


def getpbar(partial, total=None, wid=32, common_char='\u2588',
            upper_char='\u2584', lower_char='\u2580'):
    """
    Return a nice text/unicode progress bar showing
    partial and total progress

    Parameters
    ----------
    partial : float
        Partial progress expressed as decimal value.
    total : float, optional
        Total progress expresses as decimal value.
        If it is not provided or it is None, than
        partial progress will be shown as total progress.
    wid : TYPE, optional
        Width in charachters of the progress bar.
        The default is 32.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    wid -= 2
    prog = int((wid)*partial)
    if total is None:
        total_prog = prog
        common_prog = prog
    else:
        total_prog = int((wid)*total)
        common_prog = min(total_prog, prog)
    pbar_full = common_char*common_prog
    pbar_full += upper_char*(total_prog - common_prog)
    pbar_full += lower_char*(prog - common_prog)
    return (f"\u2595{{:<{wid}}}\u258F").format(pbar_full)


class SaturatedLinear1D(Fittable1DModel):
    """
    Saturated linear model.

    Parameters
    ----------
    """
    n_inputs = 1
    n_outputs = 1

    slope = Parameter()
    inter = Parameter()
    x_sat = Parameter()

    def __init__(self, slope=1, inter=0, x_sat=0, **kwargs):
        super().__init__(slope, inter, x_sat, **kwargs)

    @staticmethod
    def evaluate(x, slope, inter, x_sat):
        y_mask = x >= x_sat
        y = (x*slope + inter) * y_mask
        y += (x_sat*slope + inter) * ~y_mask
        return y


def multiple_formatter(denominator=4, number=np.pi, latex='\\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int64(np.rint(den*x/number))
        com = gcd(num, den)
        (num, den) = (int(num/com), int(den/com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return rf'${latex}$'
            elif num == -1:
                return rf'$-{latex}$'
            else:
                return rf'${num}{latex}$'
        else:
            if num == 1:
                return rf'$\frac{{{latex}}}{{{den}}}$'
            elif num == -1:
                return rf'$\frac{{-{latex}}}{{{den}}}$'
            else:
                return rf'$\frac{{{num}{latex}}}{{{den}}}$'
    return _multiple_formatter


def getvclip(img, vclip=0.5):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    vclip : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    vmin : TYPE
        DESCRIPTION.
    vmax : TYPE
        DESCRIPTION.

    """
    vmin = np.ma.median(img) - vclip*np.ma.std(img)
    vmax = np.ma.median(img) + vclip*np.ma.std(img)
    return vmin, vmax


def getlogimg(img, vclip=0.5):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    vclip : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    log_img = np.log10(1 + img - img.min())
    return log_img, *getvclip(log_img)


def makecutout(img, center, size):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    center : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    cutout = img[
        int(center[0]-size[0]/2):int(center[0]+size[0]/2),
        int(center[1]-size[1]/2):int(center[1]+size[1]/2),
    ]
    return cutout.copy()


def linear2polar(img, center=None, radius=None, output_shape=None, order=1):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    o : TYPE, optional
        DESCRIPTION. The default is None.
    radius : TYPE, optional
        DESCRIPTION. The default is None.
    output_shape : TYPE, optional
        DESCRIPTION. The default is None.
    order : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    if center is None:
        center = np.array(img.shape[:2])/2 - 0.5

    if radius is None:
        radius = (np.array(img.shape[:2])**2).sum()**0.5/2

    if output_shape is None:
        shp = 360, int(radius)
        output = np.zeros(shp, dtype=img.dtype)
    else:
        output = np.zeros(output_shape, dtype=img.dtype)
    out_h, out_w = output.shape
    rs = np.linspace(0, radius, out_w, endpoint=False)
    ts = np.linspace(0, np.pi*2, out_h, endpoint=False)
    ys = rs[:, None] * np.cos(ts) + center[1]
    xs = rs[:, None] * np.sin(ts) + center[0]
    map_coordinates(img, (xs, ys), order=order, output=output.T)
    return output


def polar2linear(img, center=None, radius=None, output_shape=None, order=1):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    center : TYPE, optional
        DESCRIPTION. The default is None.
    radius : TYPE, optional
        DESCRIPTION. The default is None.
    output_shape : TYPE, optional
        DESCRIPTION. The default is None.
    order : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    if radius is None:
        radius = img.shape[0]

    if output_shape is None:
        output = np.zeros((radius*2, radius*2), dtype=img.dtype)
    else:
        output = np.zeros(output_shape, dtype=img.dtype)

    if center is None:
        center = np.array(output.shape)/2 - 0.5
    else:
        center = np.array(center)

    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w] - center[:, None, None]
    rs = (ys**2+xs**2)**0.5
    mask_rs_zero = rs != 0
    ts = np.zeros_like(xs)
    ts[mask_rs_zero] = np.arccos(xs[mask_rs_zero]/rs[mask_rs_zero])
    ts[ys < 0] = np.pi*2 - ts[ys < 0]
    ts *= (img.shape[0]-1)/(np.pi*2)
    map_coordinates(img, (ts, rs), order=order, output=output)
    return output


def resample(arr, new_shape, strategy=np.ma.median):
    orig_shape = arr.shape
    if len(orig_shape) != len(new_shape):
        raise ValueError(
            "The resampled arry must have the same dimension as the "
            "original array!"
        )
    ratios = np.divide(orig_shape, new_shape)

    ranges = [
        np.arange(x) for x in orig_shape
    ]

    meshgrids = np.meshgrid(*ranges, indexing='ij')

    indexes = np.zeros_like(orig_shape)

    new_array = np.zeros(new_shape)

    completed = False
    while not completed:
        mask = np.ones_like(arr, dtype=bool)
        for j, ind in enumerate(indexes):
            if ind >= new_shape[j]:
                ind = 0
                indexes[j] = 0
                try:
                    indexes[j + 1] += 1
                except IndexError:
                    completed = True
                    break
            delta = ratios[j]
            mask &= meshgrids[j] >= ind*delta
            mask &= meshgrids[j] < (ind+1)*delta
        if not completed:
            new_array[tuple(indexes)] = strategy(arr[mask])
            indexes[0] += 1
    return new_array


def hough_space_from_points(xdata, ydata, aux=None, shape=(300, 300),
                            theta_start=0, theta_end=np.pi):
    """
    Compute the Hough transform for a given set of points.
    https://en.wikipedia.org/wiki/Hough_transform

    Parameters
    ----------
    xdata : TYPE
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.
    shape : TYPE, optional
        DESCRIPTION. The default is (360, 300).

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if len(xdata) != len(ydata):
        raise ValueError("xdata and ydata must have the same lenght!")

    hough_space = np.zeros(shape, dtype='float32')

    x_min = np.nanmin(xdata)
    y_min = np.nanmin(ydata)
    x_max = np.nanmax(xdata)
    y_max = np.nanmax(ydata)

    # The maximum possible value for r
    r_max = ((x_max - x_min)**2 + (y_max - y_min)**2)**0.5

    for theta in range(shape[1]):
        for i, (x_p, y_p) in enumerate(zip(xdata, ydata)):
            x = x_p - x_min
            y = y_p - y_min
            theta_rad = (theta_end - theta_start) * theta
            theta_rad += theta_start
            theta_rad /= shape[1]
            r = x*np.cos(theta_rad) + y*np.sin(theta_rad) + r_max
            r *= shape[0] / (2 * r_max)
            if aux is None:
                hough_space[int(r), int(theta)] += 1
            else:
                hough_space[int(r), int(theta)] += aux[i]
    extents = (theta_start, theta_end, -r_max, r_max)
    return hough_space, extents
