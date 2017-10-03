#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orcs.py

## Copyright (c) 2010-2017 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORCS
##
## ORCS is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORCS is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORCS.  If not, see <http://www.gnu.org/licenses/>.


"""
Utils module contains core functions that are used by the processing
classes of ORCS
"""
# import ORB
import orb.utils.spectrum
import orb.fit

import numpy as np
import logging
import warnings
import gvar

def fit_lines_in_spectrum(params, inputparams, fit_tol, spectrum,
                          theta_orig, snr_guess=None, **kwargs):
    """Basic wrapping function for spectrum fitting.

    :param params: HDFCube.params dictionary

    :param inputparams: orb.fit.InputParams instance.

    :param fit_tol: fit tolerance.
    
    :param spectrum: The spectrum to fit (1d vector).

    :param theta_orig: Original value of the incident angle in degree.

    :param snr_guess: Guess on the SNR of the spectrum. Necessary
      to make a Bayesian fit (If unknown you can set it to 'auto'
      to try an automatic mode, two fits are made - one with a
      predefined SNR and the other with the SNR deduced from the
      first fit). If None a classical fit is made.

    :param kwargs: (Optional) Model parameters that must be
      changed in the InputParams instance.    
    """
    kwargs_orig = dict(kwargs)
    
    # check snr guess param
    auto_mode = False
    bad_snr_param = False
    if snr_guess is not None:
        if isinstance(snr_guess, str):
            if snr_guess.lower() == 'auto':
                auto_mode = True
                if np.any(gvar.sdev(spectrum) != 0.):
                    spectrum_snr = gvar.mean(spectrum) / gvar.sdev(spectrum)
                    spectrum_snr[np.isinf(spectrum_snr)] = np.nan
                    snr_guess = np.nanmax(spectrum_snr)
                    logging.debug('first SNR guess computed from spectrum uncertainty: {}'.format(snr_guess))
                else:
                    snr_guess= 30
            elif snr_guess.lower() == 'none':
                snr_guess = None
                auto_mode = False
            else: bad_snr_param = True
        elif isinstance(snr_guess, bool):
                bad_snr_param = True
        elif not (isinstance(snr_guess, float)
                  or isinstance(snr_guess, int)):
            bad_snr_param = True

    logging.debug('SNR guess: {}'.format(snr_guess))


    # recompute the fwhm guess
    if 'fwhm_guess' in kwargs:
        raise ValueError('fwhm_guess must not be in kwargs. It must be set via theta_orig.')
    
    fwhm_guess_cm1 = orb.utils.spectrum.compute_line_fwhm(
        params['step_nb'] - params['zpd_index'],
        params['step'], params['order'],
        orb.utils.spectrum.theta2corr(theta_orig),
        wavenumber=params['wavenumber'])

    kwargs['fwhm_guess'] = fwhm_guess_cm1

    logging.debug('recomputed fwhm guess: {}'.format(kwargs['fwhm_guess']))

    if bad_snr_param:
        raise ValueError("snr_guess parameter not understood. It can be set to a float, 'auto' or None.")

    try:
        warnings.simplefilter('ignore')
        _fit = orb.fit._fit_lines_in_spectrum(
            spectrum, inputparams,
            fit_tol = fit_tol,
            compute_mcmc_error=False,
            snr_guess=snr_guess,
            **kwargs)
        warnings.simplefilter('default')

    except Exception, e:
        warnings.warn('Exception occured during fit: {}'.format(e))
        import traceback
        print traceback.format_exc()

        return []

    if auto_mode and _fit != []:
        snr_guess = np.nanmax(gvar.mean(spectrum)) / np.nanstd(gvar.mean(spectrum) - _fit['fitted_vector'])
        return fit_lines_in_spectrum(params, inputparams, fit_tol, spectrum,
                                     theta_orig, snr_guess=snr_guess,
                                     **kwargs_orig)
    else:
        return _fit
