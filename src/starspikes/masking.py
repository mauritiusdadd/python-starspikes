#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:21:12 2022.

@author: daddona
"""
import os
import sys
import logging
import argparse

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, rotate
from astropy.modeling import fitting

import regions
import astropy.units as u
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from . import zeropoints
from . import utils


def findobjectcenter(img, approx_star_center, cutout_size=128,
                     gauss_filter_sigma=2):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    approx_star_center : TYPE
        DESCRIPTION.
    cutout_size : TYPE, optional
        DESCRIPTION. The default is 128.
    gauss_filter_sigma : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    actual_center : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.

    """
    approx_star_center = np.array(approx_star_center)
    approx_cutout_size = np.array((cutout_size, cutout_size))

    cutout = utils.makecutout(img, approx_star_center, approx_cutout_size)
    cutout_log, cut_log_vmin, cut_log_vmax = utils.getlogimg(cutout)

    smoothed = gaussian_filter(cutout, gauss_filter_sigma)
    smoothed = np.ma.array(smoothed, mask=np.isnan(smoothed))

    smooth_vmin, smooth_vmax = utils.getvclip(smoothed)
    smoothed_log, smooth_log_vmin, smooth_log_vmax = utils.getlogimg(smoothed)

    rel_center = np.array(np.unravel_index(
        np.ma.argmax(smoothed),
        smoothed.shape
    ))

    actual_center = approx_star_center + rel_center - approx_cutout_size/2
    actual_center = np.array(actual_center)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    ax1, ax2, ax3, ax4 = np.ravel(axs)

    _ = ax1.imshow(
        cutout_log, origin='lower', cmap='gray',
        vmin=smooth_log_vmin, vmax=smooth_log_vmax
    )
    _ = ax2.imshow(cutout, origin='lower', cmap='jet')

    _ = ax3.imshow(
        smoothed_log, origin='lower', cmap='gray',
        vmin=smooth_log_vmin, vmax=smooth_log_vmax
    )
    _ = ax4.imshow(smoothed, origin='lower', cmap='jet')

    _ = ax1.set_title(
        f"Selected star ({actual_center[1]:.0f}px, {actual_center[0]:.0f}px)"
    )
    _ = ax2.set_title("Selected star (colomap)")
    _ = ax3.set_title("Selected star (smoohted)")
    _ = ax4.set_title("Selected star (smoothed colomap)")

    _ = ax4.axvline(rel_center[1], color='magenta', ls='--', lw=1)
    _ = ax4.axhline(rel_center[0], color='magenta', ls='--', lw=1)

    _ = ax2.axvline(rel_center[1], color='magenta', ls='--', lw=1)
    _ = ax2.axhline(rel_center[0], color='magenta', ls='--', lw=1)

    return actual_center, fig


def getfoldedmodel(star_cutout, n_folds=4, angular_samples=360,
                   stack_strategy='max', max_radius=None):
    """
    .

    Parameters
    ----------
    star_cutout : TYPE
        DESCRIPTION.
    n_fold : TYPE
        DESCRIPTION.
    angular_samples : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if stack_strategy == 'max':
        stackfunc = np.nanmax
    elif stack_strategy == 'min':
        stackfunc = np.nanmin

    cutout_size = min(star_cutout.shape)
    if max_radius is None:
        polar_radius = int(cutout_size/2.0)
    else:
        polar_radius = max_radius
    folding_angle = int(angular_samples/n_folds)

    try:
        mask = star_cutout.mask
    except AttributeError:
        mask = np.isnan(star_cutout)
    else:
        mask |= np.isnan(star_cutout)
    star_cutout = np.ma.array(star_cutout, mask=mask)

    cutout_polar = utils.linear2polar(
        star_cutout,
        radius=polar_radius,
        output_shape=(angular_samples, polar_radius)
    )

    # NOTE: masks are not preserved by the wrap_polar function, we need to
    #       transofrm them separately and then to apply them back to the image
    cutout_polar_mask = utils.linear2polar(
        star_cutout.mask,
        radius=polar_radius,
        output_shape=(angular_samples, polar_radius)
    )

    cutout_polar = np.ma.array(cutout_polar, mask=cutout_polar_mask)

    # Slice polar image into n_spikes stripes. The idea is that, if there are
    # diffraction spikes, each of them will be in a different stripe and in
    # the same relative position in the stripe
    polar_slices = [
        cutout_polar[folding_angle*i:folding_angle*(i+1), ...].copy()
        for i in range(n_folds)
    ]

    # Stack the stripes using the maximum value. This will preserve only
    # image components that are present in all the stripes (i.e. spikes)
    folded_cutout_polar = np.ma.array(
        stackfunc([p for p in polar_slices], axis=0),
        mask=np.min([p.mask for p in polar_slices], axis=0)
    )

    # Create a basic star model by unfolding and unwrapping the stacked stripes
    star_model = utils.polar2linear(
        np.tile(folded_cutout_polar, (n_folds, 1)),
        output_shape=star_cutout.shape
    )
    star_model = np.ma.array(star_model, mask=star_cutout.mask)

    # Compute the mean radial profile from the folded stripes
    r_profile_ydata = np.ma.mean(folded_cutout_polar, axis=0)

    # Subtract the radial profile from the folded stripes.
    # This will effectively remove the central souce and the background,
    # leaving only the diffraction spikes (if present) and other artifacts.
    folded_residual = folded_cutout_polar - r_profile_ydata

    # Create a basic star model by unfolding and unwrapping the stacked stripes
    spikes_model = utils.polar2linear(
        np.tile(folded_residual, (n_folds, 1)),
        output_shape=star_cutout.shape
    )
    spikes_model = np.ma.array(spikes_model, mask=star_cutout.mask)

    return star_model, folded_cutout_polar


def detectspikes(star_cutout, n_spikes=4, threshold=1.0e-1,
                 angular_samples=360, gauss_filter_size=1,
                 light_radius_perc=0.97):
    """
    .

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    star_center : TYPE
        DESCRIPTION.
    cutout_size : TYPE, optional
        DESCRIPTION. The default is 512.
    n_spikes : TYPE, optional
        DESCRIPTION. The default is 4.
    threshold : TYPE, optional
        DESCRIPTION. The default is 2.5e-3.
    angular_samples : TYPE, optional
        DESCRIPTION. The default is 360.

    Returns
    -------
    detectd_spikes : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.

    """
    cutout_size = np.min(star_cutout.shape)

    polar_radius = int(cutout_size/2.0)
    folding_angle = int(angular_samples/n_spikes)

    vmin, vmax = utils.getvclip(star_cutout)

    cutout_polar = utils.linear2polar(
        star_cutout,
        radius=polar_radius,
        output_shape=(angular_samples, polar_radius)
    )

    # NOTE: masks are not preserved by the wrap_polar function, we need to
    #       transofrm them separately and then to apply them back to the image
    cutout_polar_mask = utils.linear2polar(
        star_cutout.mask,
        radius=polar_radius,
        output_shape=(angular_samples, polar_radius)
    )

    cutout_polar = np.ma.array(cutout_polar, mask=cutout_polar_mask)

    # Slice polar image into n_spikes stripes. The idea is that, if there are
    # diffraction spikes, each of them will be in a different stripe and in
    # the same relative position in the stripe
    polar_slices = np.ma.array([
        cutout_polar[folding_angle*i:folding_angle*(i+1), ...].copy()
        for i in range(n_spikes)
    ])

    # Stack the stripes using the median value. This will preserve only
    # image components that are present in all the stripes (i.e. spikes)
    folded_cutout_polar = np.ma.median(polar_slices, axis=0)

    # Compute the mean radial profile from the folded stripes
    r_profile_xdata = np.arange(folded_cutout_polar.shape[1])
    r_profile_ydata = np.ma.median(cutout_polar, axis=0)

    r_profile_delta = np.nanmax(r_profile_ydata) - np.nanmin(r_profile_ydata)
    core_size_thresh = r_profile_delta * light_radius_perc
    core_size_thresh += np.nanmin(r_profile_ydata)
    core_size_mask = r_profile_ydata < core_size_thresh
    core_size = r_profile_xdata[~core_size_mask][0]

    # Compute a nomralized radial profile
    r_profile_ynorm = np.ma.max(r_profile_ydata) - r_profile_ydata
    r_profile_ynorm /= np.ma.max(r_profile_ynorm)

    # Subtract the radial profile from the folded stripes.
    # This will effectively remove the central souce and the background,
    # leaving only the diffraction spikes (if present) and other artifacts.
    folded_residual = folded_cutout_polar - r_profile_ydata

    # Divide by the normalized radial profile. This will give less importance
    # to artifacts near the center of the source that can appear due to
    # saturation
    folded_residual /= 1 + r_profile_ynorm
    folded_residual = gaussian_filter(folded_residual, sigma=gauss_filter_size)

    # Compute the mean angular profile. If spikes are present, they will appear
    # as a negative peak in the angular profile (negative because we are using
    # magnitudes)
    ang_profile_xdata = np.arange(folded_residual.shape[0], dtype='float')
    ang_profile_xdata *= 360.0/angular_samples
    ang_profile_ydata = np.ma.median(folded_residual, axis=1)
    ang_profile_ydata = gaussian_filter(
        ang_profile_ydata,
        sigma=3*360/angular_samples
    )

    img_cmap = 'gray'

    # Detect peaks in the mean angular profile. The treshold is important
    # because it permits to discard small peaks caused by noise
    delta_peak = np.nanmax(ang_profile_ydata) - np.nanmin(ang_profile_ydata)
    peaks_ind = find_peaks(
        -ang_profile_ydata,
        prominence=threshold * delta_peak
    )[0]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = np.ravel(axs)

    _ = ax1.imshow(
        star_cutout,
        origin='lower',
        aspect='auto',
        cmap=img_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax1.set_title("Star cutout")
    ax1.set_ylabel("Y [pixels]")
    ax1.set_xlabel("X [pixels]")

    _ = ax2.imshow(
        cutout_polar,
        origin='lower',
        aspect='auto',
        cmap=img_cmap,
        vmin=vmin,
        vmax=vmax
    )

    for i in range(n_spikes):
        ax2.axhline(folding_angle*i, ls='--', lw=2, alpha=0.5, color='magenta')

    ax2.set_title("Log-Polar projection")
    ax2.set_ylabel("angle [deg]")
    ax2.set_xlabel("distance from center [pixels]")

    _ = ax3.imshow(
        folded_cutout_polar,
        origin='lower',
        aspect='auto',
        cmap=img_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax3.set_title(f"{n_spikes}-folded polar projection")
    ax3.set_ylabel("angle [deg]")
    ax3.set_xlabel("distance from center [pixels]")

    _ = ax4.imshow(
        folded_residual,
        origin='lower',
        aspect='auto',
        cmap='jet',
        vmin=np.ma.min(ang_profile_ydata)/2,
        vmax=np.ma.max(ang_profile_ydata)*2
    )

    ax4.set_title(f"{n_spikes}-folded polar projection residuals")
    ax4.set_ylabel("angle [deg]")
    ax4.set_xlabel("distance from center [pixels]")

    _ = ax5.plot(
        r_profile_xdata,
        r_profile_ydata,
        lw=3,
        label='Measured profile'
    )

    # ax5.axvline(core_size, color='magenta', ls='--', lw=2)

    ax5.set_xlim(1, np.max(r_profile_xdata))

    ax5.set_title(f"{n_spikes}-folded median radial profile")
    ax5.set_ylabel("Magnitude")
    ax5.set_xlabel("distance from center [pixels]")
    ax5.invert_yaxis()

    _ = ax5.legend()

    _ = ax6.plot(
        ang_profile_ydata,
        ang_profile_xdata,
        lw=3,
        label='Measured profile'
    )

    _ = ax6.axvline(
        np.ma.median(ang_profile_ydata),
        lw=2,
        color='red',
        alpha=0.5
    )

    detectd_spikes = []
    for p_id in peaks_ind:
        px = ang_profile_xdata[p_id]
        py = ang_profile_ydata[p_id]
        detectd_spikes.append((px, -py, core_size, r_profile_delta, n_spikes))
        _ = ax6.axhline(px, lw=2, ls='--', color='magenta')

    ax6.set_title(f"{n_spikes}-folded median angular profile")
    ax6.set_xlabel("Magnitude")
    ax6.set_ylabel("angle [degrees]")
    ax6.set_xlim(
        np.ma.min(ang_profile_ydata)*2,
        np.ma.max(ang_profile_ydata)*2
    )

    plt.tight_layout()
    return detectd_spikes, fig


def getspikesize(cutout, spike_data):
    rotated = rotate(cutout, spike_data['ANGLE'] + 45)
    star_model, spike_model = getfoldedmodel(
        rotated,
        n_folds=spike_data['N_SPIKES'],
        stack_strategy='max',
        max_radius=int(np.min(cutout.shape[0:2])/2 - 1)
    )

    spike_model = utils.resample(
        spike_model,
        [3, int(spike_model.shape[1]/10)]
    )

    background = (spike_model[0, ...] + spike_model[2, ...])/2

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    ax1, ax2, ax3 = np.ravel(axs)

    ax1.imshow(
        spike_model,
        aspect='auto',
        cmap='jet'
    )

    ax2.plot(spike_model[1, ...], label='Star + Spike profile')
    ax2.plot(background, label='Star profile')
    ax2.legend()

    mask = spike_model[1, ...] > background
    mask[:int(spike_data['CORE_SIZE'])] = 0

    ax3.plot(mask)

    size = np.argmax(mask)*10

    if not mask.sum():
        return fig, None
    else:
        return fig, size


def getmagpartition(catalog, figsize=(20, 15),
                    magauto_key='MAG_AUTO', mumax_key='MU_MAX',
                    star_key='CLASS_STAR', max_mag_treshold=40):

    catalog = catalog[catalog[magauto_key] < max_mag_treshold]

    hough_diagram, hough_extents = utils.hough_space_from_points(
        xdata=catalog[magauto_key],
        ydata=catalog[mumax_key],
    )

    hough_max = np.unravel_index(hough_diagram.argmax(), hough_diagram.shape)

    theta = hough_extents[1] * hough_max[1] / hough_diagram.shape[1]
    dist = hough_max[0] / hough_diagram.shape[0]
    dist *= (hough_extents[3] - hough_extents[2])
    dist += hough_extents[2]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(
        hough_diagram,
        origin='lower',
        extent=hough_extents,
        aspect='auto',
        cmap='jet'
    )

    ax.axvline(theta, c='magenta', ls='--', lw=2)
    ax.axhline(dist, c='magenta', ls='--', lw=2)
    ax.set_xlabel("angle [rad]")
    ax.set_ylabel("distance [mag]")
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(utils.multiple_formatter()))

    theta += np.pi / 2

    # initialize a linear fitter
    fitter = fitting.LevMarLSQFitter()

    mymodel = utils.SaturatedLinear1D(
        slope=np.tan(theta),
        inter=0,
        x_sat=np.nanmean(catalog[magauto_key]),
    )

    # fit the data with the fitter
    fitted_line = fitter(
        model=mymodel,
        x=catalog[magauto_key],
        y=catalog[mumax_key],
        maxiter=8000,
        epsilon=1e-14,
        estimate_jacobian=True,
    )

    if fitted_line.x_sat.value < np.min(catalog[magauto_key]):
        x_sat = np.min(catalog[magauto_key])
    else:
        x_sat = fitted_line.x_sat.value

    params = {
        'slope': fitted_line.slope.value,
        'intercept': fitted_line.inter.value,
        'sat_mag_auto': x_sat,
        'sat_mu_max': fitted_line.x_sat*fitted_line.slope + fitted_line.inter
    }

    return params, ax, fig


def distanceFromLine(x_coords, y_coords, slope, intercept):
    dist = np.abs(slope * x_coords + intercept - y_coords)
    dist = dist / np.sqrt(1 + slope**2)
    return dist


def __argsHandler(options=None):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'input_image', metavar='INPUT_IMG', type=str, nargs=1,
        help='The input image in fits format'
    )

    parser.add_argument(
        'input_catalog', metavar='INPUT_CAT', type=str, nargs=1,
        help='A catalog generated by sectractor using the same input image. '
        'Must contain the columns X_IMAGE, Y_MAGE, MAG_AUTO and MU_MAX.'
    )

    parser.add_argument(
        '--img-hdu', metavar='IMG_HDU', type=int, default=1,
        help='The HDU containing the image data to use. If this argument '
        'is not specified, the image is loaded from the first HDU.'
    )

    parser.add_argument(
        '--cat-hdu', metavar='CAT_HDU', type=int, default=1,
        help='The HDU containing the catalog to use (eg. for LDAC fits '
        'catalog generated by sextractor use %(metavar)s=2). If this argument '
        'is not specified, the image is loaded from the first HDU.'
    )

    parser.add_argument(
        '--check-images', default=False, action='store_true',
        help='Generate control images for the various operations performed.'
    )

    parser.add_argument(
        '--check-dir', metavar='CHK_FOLDER', type=str, default='checkimgs',
        nargs=1, help='Set the directory where check images and files are '
        'written. %(metavar)s should be a relative or an absolute path, if the'
        ' folder does not exists it will be automatically created. '
        'If this argument is not provided, the default value '
        '%(metavar)s=%(default)s is used.'
    )

    parser.add_argument(
        '--out-dir', metavar='OUT_FOLDER', type=str, default='.',
        help='Set the output directory where cleaned catalog are saved. '
        'If not specified, the current working directory is used.'
    )
    parser.add_argument(
        '--check-dpi', metavar='DPI', type=int, default=150,
        help='Set the DPI of the control images. if this parameter is not '
        'specified then the default value %(metavar)s=%(default)d is used.'
    )

    parser.add_argument(
        '--figsize', metavar='FIG_SIZE', type=str,
        default='20x15', help='Set the size in inches of the control '
        'images created by the program. The format of this parameter should '
        'be WIDTHxHEIGHT (for example 10x5). If this argument is not '
        'specified, the default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--cutout-size', metavar='CUTOUT_SIZE', type=int, default=64,
        help='Set the size of the cutouts, centred on each star, that will '
        'be used to detect diffraction spikes and to compute the size of the '
        'masks around them. If it is not specified, this parameter will have a'
        'default value of %(metavar)s=%(default)d.'
    )

    parser.add_argument(
        '--angle-tol', metavar='ANGLE_TOL', type=float, default=2,
        help='Set the max difference, in degrees, between two angles to be '
        'considered equal. If it is not specified, this parameter will have a'
        'default value of %(metavar)s=%(default)d.'
    )

    parser.add_argument(
        '--dist-tol', metavar='DIST_TOL', type=float, default=5,
        help='Set the max distance, in pixels, between two objects to be '
        'considered the same. If it is not specified, this parameter will have'
        ' a default value of %(metavar)s=%(default)d.'
    )

    parser.add_argument(
        '--magauto-thresh', metavar='MAG_AUTO_THRESH', type=int, default=50,
        nargs=1, help='Ignore objects with MAG_AUTO > %(metavar)s. If not '
        'specified, the deafult value %(metavar)s=%(default)d is used.'
    )

    parser.add_argument(
        '--magauto-key', metavar='MAGAUTO_KEY', type=str, default='MAG_AUTO',
        nargs=1, help='Set the name of the column containg the automatic '
        'magnitudes. For catalogs generated by sextractor '
        '%(metavar)s=MAG_AUTO. If this argument is not specified the default '
        'value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--mumax-key', metavar='MUMAX_KEY', type=str, default='MU_MAX',
        nargs=1, help='Set the name of the column containg the peak surface '
        'brightness above background. For catalogs generated by sextractor '
        '%(metavar)s=MU_MAX. If this argument is not specified the default '
        'value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--class-key', metavar='CLASS_STAR_KEY', type=str,
        default='CLASS_STAR', help='Set the name of the column '
        'containg class of the objects. For catalogs generated by sextractor '
        '%(metavar)s=CLASS_STAR. If this argument is not specified the default'
        ' value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--ximage-key', metavar='X_IMAGE_KEY', type=str, default='X_IMAGE',
        nargs=1, help='Set the name of the column containg the x pixel'
        'coordinates of the objects. For catalogs generated by sextractor '
        '%(metavar)s=%(default)s. If this argument is not specified the '
        'default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--yimage-key', metavar='Y_IMAGE_KEY', type=str, default='Y_IMAGE',
        nargs=1, help='Set the name of the column containg the y pixel'
        'coordinates of the objects. For catalogs generated by sextractor '
        '%(metavar)s=%(default)s. If this argument is not specified the '
        'default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--aimage-key', metavar='A_IMAGE_KEY', type=str, default='A_IMAGE',
        nargs=1, help='Set the name of the column containg the major axis'
        'of the objects. For catalogs generated by sextractor '
        '%(metavar)s=%(default)s. If this argument is not specified the '
        'default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--bimage-key', metavar='B_IMAGE_KEY', type=str, default='B_IMAGE',
        nargs=1, help='Set the name of the column containg the y pixel'
        'coordinates of the objects. For catalogs generated by sextractor '
        '%(metavar)s=%(default)s. If this argument is not specified the '
        'default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--thetaimage-key', metavar='THETA_IMAGE_KEY', type=str,
        default='THETA_IMAGE', nargs=1, help='Set the name of the column '
        'containg the rotation angle of the major axis of the objects relative'
        'to the first image axis. For catalogs generated by sextractor '
        '%(metavar)s=%(default)s. If this argument is not specified the '
        'default value %(metavar)s=%(default)s is used'
    )

    parser.add_argument(
        '--selection-a', metavar='SELE_PARAM_A', type=float,
        default=None, help='A parameter used in the selection of the '
        'brightest and saturated objects. It can be any real number. If this '
        'parameter is not specified, then it is computed automatically by the '
        'program. For further information please refer to the program '
        'documentation.'
    )

    parser.add_argument(
        '--mag-delta', metavar='SELE_PARAM_B', type=float,
        default=2, help='A parameter used in the selection of the '
        'brightest and saturated objects. It can be any real number. If this '
        'parameter is not specified, then it is computed automatically by the '
        'program. For further information please refer to the program '
        'documentation.'
    )

    parser.add_argument(
        '--ellipticity-threshold', metavar='TRESHOLD', type=float, default=0.2,
        help='Sets the ellipticity threshold below which objects that are '
        'maked as spurios spike detections will be removed.'
    )

    parser.add_argument(
        '--n-spikes', metavar='N', type=int, default=4,
        help='Set the number of spikes a star has. The default value is .'
        '%(metavar)s=%(default)s'
    )

    parser.add_argument(
        '--mask-width', metavar='WIDTH_PIXELS', type=int, default=None,
        help='Set the width of the mask around the spikes. If this parameter '
        'is not specified the default value %(metavar)s=%(default)s is used.'
    )

    parser.add_argument(
        '--verbose', '-v', default=False, action='store_true',
        help='Increase the information outputed by the program.'
    )

    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='Increase further the information outputed by the program.'
    )

    return parser.parse_args(options)


def main(options=None):
    """
    Run the main program.

    Returns
    -------
    None.

    """
    font = {
        'size': 12
    }
    mpl.rc('font', **font)

    args = __argsHandler(options)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()

    if args.verbose or args.debug:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARN)

    logger.addHandler(console_handler)

    star_cutout_size = int(args.cutout_size)
    angle_tol_deg = float(args.angle_tol)

    basename = os.path.basename(args.input_image[0])
    basename = os.path.splitext(basename)[0]

    cat_basename = os.path.basename(args.input_catalog[0])
    cat_basename = os.path.splitext(cat_basename)[0]

    if not os.path.isdir(args.check_dir):
        os.mkdir(args.check_dir)

    chk_img_out = os.path.join(
        args.check_dir,
        basename
    )

    if not os.path.isdir(chk_img_out):
        os.mkdir(chk_img_out)

    logging.info(f"Loading input catalog {args.input_catalog[0]}...")
    try:
        cat = Table.read(args.input_catalog[0], hdu=args.cat_hdu)
    except FileNotFoundError as exc:
        logging.error(f"Invalid catalog: {str(exc)}")
        sys.exit(1)
    else:
        cat = cat[cat[args.magauto_key] < args.magauto_thresh]

    logging.info(f"Loading image {args.input_image[0]}")
    try:
        img_header = fits.getheader(args.input_image[0])
    except FileNotFoundError as exc:
        logging.error(f"Invalid fits image: {str(exc)}")
        sys.exit(1)

    zpt_info = zeropoints.getZeroPointInfo(args.input_image[0])

    img = fits.getdata(args.input_image[0], ext=args.img_hdu-1)
    img = np.ma.array(img, mask=np.isnan(img))
    img_wcs = WCS(img_header)

    img_height, img_width = img.shape[0:2]
    img_figsize = (
        img_width / float(args.check_dpi),
        img_height / float(args.check_dpi)
    )

    vmin, vmax = utils.getvclip(img)

    # log_img, log_vmin, log_vmax = getlogimg(img)
    # Compute the image in magnitude units
    # NOTE: we use ACS and WFC3/IR images, that are in e- units and are
    #       images are already corrected for the instrumental gain, so the
    #       conversion constant to pass from count to flux is just the ratio
    #       between PHOTFLAM and EXPTIME.
    #
    #       Also note that negative counts (or fluxes) have no sense!
    #       For more information take a look at:
    #       https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry
    masked_img = np.ma.array(img, mask=img == np.nanmin(img))
    log_img = -2.5 * np.ma.log10(masked_img - np.ma.min(masked_img))
    log_img += zpt_info['zero_point']
    log_img = np.ma.array(log_img, mask=np.isnan(log_img) | img.mask)

    log_vmin, log_vmax = utils.getvclip(log_img)

    mean, median, std = sigma_clipped_stats(img, sigma=1.0)

    logging.info(
        "Extracting brightest and saturated stars from input catalog..."
    )
    # Detecting stars

    class_star_thresh = 0.8

    only_stars = cat[cat[args.class_key] >= class_star_thresh]
    params, ax, fig = getmagpartition(
        only_stars,
        magauto_key=args.magauto_key,
        mumax_key=args.mumax_key,
        star_key=args.class_key,
        max_mag_treshold=40,
    )

    print("", file=sys.stderr)
    logging.info(f"Slope: {params['slope']}")
    logging.info(f"Intercept: {params['intercept']}")
    logging.info(f"Saturation MAG_AUTO: {params['sat_mag_auto']}")
    logging.info(f"Saturation MU_MAX: {params['sat_mu_max']}")

    # Selecting points that are approximately  in the linear region
    # of the star sequence
    mag_hi_thresh = params['sat_mag_auto'] + args.mag_delta
    mumax_hi_thresh = mag_hi_thresh * params['slope'] + params['intercept']

    mask_1 = only_stars[args.magauto_key] >= params['sat_mag_auto']
    mask_1 &= only_stars[args.mumax_key] <= mumax_hi_thresh

    selected_stars_for_stat = only_stars[mask_1]

    distances = distanceFromLine(
        selected_stars_for_stat[args.magauto_key],
        selected_stars_for_stat[args.mumax_key],
        params['slope'],
        params['intercept']
    )

    selection_offset = 5 * np.std(distances)
    mumax_sat_thresh = params['sat_mu_max'] + selection_offset
    mumax_hi_thresh += selection_offset
    intercept_offset = selection_offset

    logging.info(f"Selection offset: {selection_offset}")
    logging.info(f"MU_MAX saturation threshold: {mumax_sat_thresh}")
    print("", file=sys.stderr)

    # Mask for selecting saturated objects from the input catalog
    saturated_mask = cat[args.mumax_key] <= mumax_sat_thresh

    # Mask for selecting brightest non saturated objects from the catalog
    mumax_thresh_values = cat[args.magauto_key] * params['slope']
    mumax_thresh_values += params['intercept'] + intercept_offset
    brightest_mask1 = cat[args.mumax_key] <= mumax_thresh_values
    brightest_mask1 &= cat[args.magauto_key] <= mag_hi_thresh

    brightest_mask2 = cat[args.magauto_key] > mag_hi_thresh
    brightest_mask2 &= cat[args.mumax_key] <= mumax_hi_thresh

    selection_mask = saturated_mask | brightest_mask1 | brightest_mask2
    brightest_objects = cat[selection_mask]

    if args.check_images:
        plt.tight_layout()
        fig.savefig(
            os.path.join(chk_img_out, "hough_space.png"),
            dpi=args.check_dpi
        )
        plt.close(fig)

        idx_sorted = np.argsort(cat[args.class_key])
        scatter_marker_size = 15
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_facecolor('#EAEAF2')
        ax.grid(
            visible=None, which='major', axis='both', color='white', alpha=1.0
        )
        ax.set_axisbelow(True)
        scatter = ax.scatter(
            cat[args.magauto_key][idx_sorted],
            cat[args.mumax_key][idx_sorted],
            c=cat[args.class_key][idx_sorted],
            cmap='plasma',
            s=scatter_marker_size,
            alpha=1.0,
            label='Objects'
        )

        _ = ax.scatter(
            brightest_objects[args.magauto_key],
            brightest_objects[args.mumax_key],
            facecolor='none',
            edgecolor='red',
            s=scatter_marker_size,
            lw=1.5,
            alpha=0.5,
            label='Selected objects'
        )

        plt.margins(0.05, 0.05)

        if args.debug:
            _ = ax.scatter(
                selected_stars_for_stat[args.magauto_key],
                selected_stars_for_stat[args.mumax_key],
                marker='+',
                s=80,
                label='Star selected for stats computation',
                zorder=3
            )
            ax.axhline(
                params['sat_mag_auto'], lw=1, ls='--', c='gray', alpha=0.5
            )
            ax.axhline(
                mag_hi_thresh, lw=1, ls='--', c='gray', alpha=0.5
            )

        ax.set_xlabel(fr"{args.magauto_key} [$mag$]")
        ax.set_ylabel(fr"{args.mumax_key} [$mag \cdot arcsec^{{-2}}$]")

        cbar = fig.colorbar(
            scatter,
            location='right',
            anchor=(0, 0.3)
        )

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(f"{args.class_key}", rotation=270)

        plt.tight_layout()

        x_min = ax.get_xlim()[0]
        x_max = ax.get_xlim()[1]

        y_max = x_max*params['slope'] + params['intercept']

        # Plotting star sequence fit
        fit_lw = 1.5
        ax.plot(
            (x_min, params['sat_mag_auto']),
            (params['sat_mu_max'], params['sat_mu_max']),
            color='magenta',
            ls='-',
            lw=fit_lw,
            alpha=0.5
        )

        ax.plot(
            (params['sat_mag_auto'], x_max),
            (
                params['sat_mu_max'],
                y_max
            ),
            color='magenta',
            ls='-',
            lw=fit_lw,
            label='Fitted star sequence',
            alpha=0.5
        )

        # Plot brightest and saturated objects selection region
        sel_boundary_lw = 1.5
        ax.plot(
            (x_min, params['sat_mag_auto']),
            (mumax_sat_thresh, mumax_sat_thresh),
            color='red',
            ls='--',
            lw=sel_boundary_lw
        )

        ax.plot(
            (params['sat_mag_auto'], mag_hi_thresh),
            (mumax_sat_thresh, mumax_hi_thresh),
            color='red',
            ls='--',
            lw=sel_boundary_lw,
        )

        ax.plot(
            (mag_hi_thresh, x_max),
            (mumax_hi_thresh, mumax_hi_thresh),
            color='red',
            ls='--',
            lw=sel_boundary_lw,
            label='Stars selection region boundary'
        )

        _ = ax.annotate(
            '$x_{sat}$',
            xy=(params['sat_mag_auto'], 0),
            xytext=(params['sat_mag_auto'], 0.025),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            fontsize=16,
            arrowprops=dict(
                facecolor='black',
                arrowstyle='-'
            )
        )

        _ = ax.legend()

        fig.savefig(
            os.path.join(chk_img_out, "brightest_star_selection.png"),
            dpi=args.check_dpi
        )
        plt.close(fig)

    logging.info(f"Selected {len(brightest_objects)} sources")

    positions = brightest_objects.copy()

    detection_chk_img_out = os.path.join(chk_img_out, "sources")
    if not os.path.isdir(detection_chk_img_out):
        os.mkdir(detection_chk_img_out)

    for i, source in enumerate(positions):
        actual_center, fig = findobjectcenter(
            img,
            (source[args.yimage_key], source[args.ximage_key]),
            cutout_size=star_cutout_size
        )
        old_center = np.zeros_like(actual_center)

        # Check if we have found the center of the objects. For very bright
        # stars more than one iteration is needed, especially for fake
        # detections along spikes
        while np.linalg.norm(actual_center - old_center) > args.dist_tol:
            plt.close(fig)
            old_center = actual_center
            actual_center, fig = findobjectcenter(
                img,
                old_center,
                cutout_size=star_cutout_size
            )

        source_id = f"{actual_center[1]:04.0f}_{actual_center[0]:04.0f}"

        plt.tight_layout()
        fig.savefig(
            os.path.join(detection_chk_img_out, f"{source_id}.png"),
            dpi=args.check_dpi
        )
        plt.close(fig)

        positions[args.ximage_key][i] = actual_center[1]
        positions[args.yimage_key][i] = actual_center[0]

    if args.check_images:
        # Show the detection result
        fig, ax = plt.subplots(
            1, 1,
            figsize=img_figsize,
            subplot_kw={'projection': img_wcs}
        )

        im = ax.imshow(
            log_img,
            origin='lower',
            cmap='gray_r',
            vmin=log_vmin,
            vmax=log_vmax,
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('AB magnitude', rotation=270)

        _ = ax.scatter(
            positions[args.ximage_key],
            positions[args.yimage_key],
            marker='o',
            lw=2,
            s=80,
            facecolor='none',
            edgecolor='red',
            alpha=1
        )

        plt.tight_layout()
        fig.savefig(
            os.path.join(chk_img_out, "brightest.png"),
            dpi=args.check_dpi
        )
        plt.close(fig)

    profiles_chk_img_out = os.path.join(chk_img_out, "sources_profiles")
    if not os.path.isdir(profiles_chk_img_out):
        os.mkdir(profiles_chk_img_out)

    valid_spikes = Table(
        names=[
            'ID',
            args.ximage_key,
            args.yimage_key,
            'ANGLE',
            'N_SPIKES',
            'AMPLITUDE',
            'CORE_SIZE',
            'SPIKE_SIZE',
            'MAG_PEAK',
            'MU_MAX',
            'MAG'
        ],
        dtype=[
            '<U12', 'float32', 'float32', 'float32', 'uint8',
            'float32', 'float32', 'float32', 'float32', 'float32', 'float32'
        ]
    )

    # NOTE: using logarithmic image to detect fainter diffraction spikes
    for i, source in enumerate(positions):
        partial_progress = (i+1) / len(positions)
        sys.stderr.write(
            utils.STD_PROGRESS_FMT.format(
                "Computing sources profiles: ",
                utils.getpbar(partial_progress, 0.3),
                partial_progress
            )
        )
        sys.stderr.flush()
        actual_center = (source[args.yimage_key], source[args.ximage_key])

        cutout_size = star_cutout_size

        # Detect objects potentially having spikes
        detectd_spikes, fig = detectspikes(
            utils.makecutout(
                log_img,
                actual_center,
                np.array((cutout_size, cutout_size)),
            ),
            n_spikes=args.n_spikes
        )

        # Maybe the cutout is too small, retry with a bigger one
        while detectd_spikes[0][2] > cutout_size/3:
            plt.close(fig)
            cutout_size *= 2
            detectd_spikes, fig = detectspikes(
                utils.makecutout(
                    log_img,
                    actual_center,
                    np.array((cutout_size, cutout_size)),
                ),
                n_spikes=args.n_spikes
            )

        source_id = f"{actual_center[1]:04.0f}_{actual_center[0]:04.0f}"

        # If there are no spikes do nothing for this source
        for spike in detectd_spikes:
            valid_spikes.add_row(
                [
                    source_id,
                    actual_center[1],
                    actual_center[0],
                    spike[0],
                    spike[4],
                    spike[1],
                    spike[2],
                    -99,
                    spike[3],
                    source[args.mumax_key],
                    source[args.magauto_key],
                ]
            )

        if args.check_images:
            plt.tight_layout()
            fig.savefig(
                os.path.join(profiles_chk_img_out, f"{source_id}.png"),
                dpi=args.check_dpi
            )
            plt.close(fig)

    logging.info("")

    valid_spikes.write(
        os.path.join(chk_img_out, "spikes_detection_ldac.fits"),
        overwrite=True
    )
    # Diffraction spikes are caused by the vanes of the spyder supporting the
    # secondary mirror of the telescope, therefore they angular position do not
    # change across the image (unless severe distrotion correction).
    #
    # For this reasone, in order to select effectively real spikes, we just
    # compute the median value of the angular position of the detected spikes,
    # then we consider as real spikes only the ones having an angular position
    # compatible with the median within +- angle_tol_deg

    median_angle = np.median(valid_spikes['ANGLE'])

    angle_mask = np.abs(valid_spikes['ANGLE'] - median_angle) <= angle_tol_deg
    valid_spikes = valid_spikes[angle_mask]
    valid_spikes['ANGLE'] = median_angle

    """
    spikes_chk_img_out = os.path.join(chk_img_out, "spikes_profiles")
    if args.check_images and not os.path.isdir(spikes_chk_img_out):
        os.mkdir(spikes_chk_img_out)

    for j, spike in enumerate(valid_spikes):
        partial_progress = (j+1) / len(valid_spikes)
        sys.stderr.write(
            utils.STD_PROGRESS_FMT.format(
                "Computing spikes length:    ",
                utils.getpbar(partial_progress, 0.6),
                partial_progress
            )
        )
        sys.stderr.flush()
        initial_cutout_size = args.cutout_size
        cutout = utils.makecutout(
            log_img,
            (
                spike[args.yimage_key],
                spike[args.ximage_key]
            ),
            (
                initial_cutout_size,
                initial_cutout_size
            )
        )

        fig, spike_size = getspikesize(cutout, spike)
        count = 10
        while spike_size is None and count > 0:
            plt.close(fig)
            initial_cutout_size *= 2
            count -= 1
            cutout = utils.makecutout(
                log_img,
                (
                    spike[args.yimage_key],
                    spike[args.ximage_key]
                ),
                (
                    initial_cutout_size,
                    initial_cutout_size
                )
            )
            fig, spike_size = getspikesize(cutout, spike)
        spike['SPIKE_SIZE'] = spike_size

        if args.check_images:
            fig.savefig(
                os.path.join(spikes_chk_img_out, f"{spike['ID']}_spike.png")
            )
        plt.close(fig)
    """
    if args.check_images:
        valid_spikes.write(
            os.path.join(chk_img_out, f"{basename}_spikes.fits"),
            overwrite=True
        )

    # Mean star profile
    if args.check_images:
        cutouts = []
        for i, source in enumerate(valid_spikes):
            actual_center = (source[args.ximage_key], source[args.yimage_key])
            star_cutout = utils.makecutout(
                img,
                actual_center,
                np.array((star_cutout_size, star_cutout_size))
            )
            bkg = np.ma.min(star_cutout)
            star_cutout -= bkg
            star_cutout /= np.ma.max(star_cutout)
            cutouts.append(star_cutout)

    if args.check_images:
        fig, ax = plt.subplots(
            1, 1,
            figsize=img_figsize,
            subplot_kw={'projection': img_wcs}
        )

        _ = ax.imshow(
            log_img,
            origin='lower',
            cmap='gray_r',
            vmin=log_vmin,
            vmax=log_vmax
        )

    compound_mask_region = None

    if args.mask_width is None:
        spike_width = 20
    else:
        spike_width = args.spike_width

    mask_list = []
    for j, spike in enumerate(valid_spikes):
        partial_progress = (j + 1)/len(valid_spikes)
        sys.stderr.write(
            utils.STD_PROGRESS_FMT.format(
                "Generating mask regions:    ",
                utils.getpbar(partial_progress, 0.8),
                partial_progress
            )
        )
        sys.stderr.flush()
        if spike['SPIKE_SIZE'] < 0:
            len_spike = 150 + 4.7e13 * (spike['MAG'] ** -9.37)
            print(len_spike)
        else:
            len_spike = 2*spike['SPIKE_SIZE']

        spike_1 = regions.RectanglePixelRegion(
            center=regions.PixCoord(spike['X_IMAGE'], spike['Y_IMAGE']),
            width=spike_width,
            height=len_spike,
            angle=spike['ANGLE'] * u.deg
        )

        spike_2 = regions.RectanglePixelRegion(
            center=regions.PixCoord(spike['X_IMAGE'], spike['Y_IMAGE']),
            width=spike_width,
            height=len_spike,
            angle=(spike['ANGLE'] + 90) * u.deg
        )

        circle = regions.CirclePixelRegion(
            center=regions.PixCoord(spike['X_IMAGE'], spike['Y_IMAGE']),
            radius=2*spike['CORE_SIZE'],
        )
        mask_list.append(spike_1)
        mask_list.append(spike_2)
        mask_list.append(circle)

    my_regions = regions.Regions(mask_list)
    my_regions.write(
        os.path.join(chk_img_out, f"{basename}_masks.reg"),
        format='ds9',
        overwrite=True
    )
    for region in mask_list:
        if compound_mask_region is None:
            compound_mask_region = region
        else:
            compound_mask_region |= region
        if args.check_images:
            region.plot(
                ax=ax,
                facecolor='red',
                edgecolor='magenta',
                lw=1.0,
                alpha=0.6
            )

    if args.check_images:
        fig.savefig(
            os.path.join(chk_img_out, f"{basename}_mask_regions.png")
        )

    coords = regions.PixCoord(
        cat[args.ximage_key], cat[args.yimage_key]
    )

    ellipticity = 1 - cat[args.bimage_key]/cat[args.aimage_key]
    high_ellipticity = ellipticity > args.ellipticity_threshold

    cleanup_mask = compound_mask_region.contains(coords) & high_ellipticity

    cleaned_catalog = cat[~cleanup_mask]
    deleted_catalog = cat[cleanup_mask]

    if args.check_images:
        for i, obj in enumerate(cleaned_catalog):
            if (i % 100) == 0:
                partial_progress = (i + 1)/len(cleaned_catalog)
                sys.stderr.write(
                    utils.STD_PROGRESS_FMT.format(
                        "Generating detection image :",
                        utils.getpbar(partial_progress, 1.0),
                        partial_progress
                    )
                )
                sys.stderr.flush()
            obj_kron = obj['KRON_RADIUS']
            ellipse = patches.Ellipse(
                (obj[args.ximage_key], obj[args.yimage_key]),
                width=obj[args.aimage_key]*obj_kron,
                height=obj[args.bimage_key]*obj_kron,
                angle=obj[args.thetaimage_key],
                facecolor='none',
                edgecolor='red',
                alpha=0.9,
                linewidth=1
            )
            ax.add_patch(ellipse)
        fig.savefig(
            os.path.join(chk_img_out, f"{basename}_src_clean.png")
        )
        plt.close(fig)

        deleted_catalog.write(
            os.path.join(chk_img_out, f"{cat_basename}_deleted.fits"),
            overwrite=True
        )

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    cleaned_catalog.write(
        os.path.join(args.out_dir, f"{cat_basename}_clean.fits"),
        overwrite=True
    )


if __name__ == '__main__':
    main([])