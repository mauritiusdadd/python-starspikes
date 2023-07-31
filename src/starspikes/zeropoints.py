#!/usr/bin/env python3
import sys
import numpy as np
from astropy.io import fits
from astropy import units


def getZeroPointInfo(filename):
    hdr = fits.getheader(filename, ext=0)
    phot_f_lam = hdr['PHOTFLAM']
    phot_p_lam = hdr['PHOTPLAM']
    exp_time = hdr['EXPTIME']

    try:
        science_units = hdr['BUNIT']
    except KeyError:
        science_units = None

    acs_zpt = -2.5 * np.log10(phot_f_lam) - 21.10
    acs_zpt += -5 * np.log10(phot_p_lam) + 18.6921

    acs_zpt_pexp = acs_zpt + 2.5 * np.log10(exp_time)
    acs_zpt_mexp = acs_zpt - 2.5 * np.log10(exp_time)

    if science_units is not None and science_units.lower().endswith('/s'):
        counts_to_flux = phot_f_lam
    else:
        counts_to_flux = phot_f_lam/exp_time

    zpt_dict = {
        "exp_time": exp_time,
        "phot_f_lam": phot_f_lam,
        "phot_p_lam": (phot_p_lam / 10) * units.nm,
        "zero_point": acs_zpt,
        "zero_point_p": acs_zpt_pexp,
        "zero_point_m": acs_zpt_mexp,
        'counts_to_flux': counts_to_flux
    }
    return zpt_dict


def printZeroPointInfo(filename):
    zpt_dict = getZeroPointInfo(filename)
    print(f"\n{filename}")
    s = "Exp time: {exp_time}\n"
    s += "Pivot wavelenght: {phot_p_lam:.0f}\n"
    s += "Zero point: {zero_point}\n"
    s += "Zero point (+m): {zero_point_p}\n"
    s += "Zero point (-m): {zero_point_m}\n"
    print(s.format(**zpt_dict))


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        printZeroPointInfo(filename)
