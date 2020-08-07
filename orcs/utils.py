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

import numpy as np
import logging
import warnings
import gvar
import scipy
import time

# import ORB
import orb.utils.log
import orb.utils.spectrum
import orb.fit

# def fit_lines_in_spectrum(params, inputparams, fit_tol, spectrum,
#                           theta_orig, snr_guess=None, max_iter=None,
#                           debug=False, **kwargs):
#     """Basic wrapping function for spectrum fitting.

#     :param params: HDFCube.params dictionary

#     :param inputparams: orb.fit.InputParams instance.

#     :param fit_tol: fit tolerance.

#     :param spectrum: The spectrum to fit (1d vector).

#     :param theta_orig: Original value of the incident angle in degree.


#     :param kwargs: (Optional) Model parameters that must be
#       changed in the InputParams instance.
#     """
#     import orb.utils.spectrum

#     kwargs_orig = dict(kwargs)
#     if debug:
#         import orb.utils.log
#         orb.utils.log.setup_socket_logging()

    
#     auto_mode = False
#                 if np.any(gvar.sdev(spectrum) != 0.):
#                     spectrum_snr = gvar.mean(spectrum) / gvar.sdev(spectrum)
#                     spectrum_snr[np.isinf(spectrum_snr)] = np.nan
#                     snr_guess = np.nanmax(spectrum_snr)
#                     logging.debug('first SNR guess computed from spectrum uncertainty: {}'.format(snr_guess))
#                 else:
#                     snr_guess= 30
#             elif snr_guess.lower() == 'none':
#                 snr_guess = None
#                 auto_mode = False
#             else: bad_snr_param = True
#         elif isinstance(snr_guess, bool):
#                 bad_snr_param = True
#         elif not (isinstance(snr_guess, float)
#                   or isinstance(snr_guess, int)):
#             bad_snr_param = True

    


    
#     if max_iter is None:
#         max_iter = max(100 * inputparams.allparams['line_nb'], 1000)

#     try:
#         warnings.simplefilter('ignore')
#         _fit = orb.fit._fit_lines_in_spectrum(
#             spectrum, inputparams,
#             fit_tol = fit_tol,
#             compute_mcmc_error=False,
#             snr_guess=snr_guess,
#             max_iter=max_iter,
#             **kwargs)
#         warnings.simplefilter('default')

#     except Exception, e:
#         logging.warning('Exception occured during fit: {}'.format(e))
#         import traceback
#         print traceback.format_exc()

#         return []

#     if auto_mode and _fit != []:
#         snr_guess = np.nanmax(gvar.mean(spectrum)) / np.nanstd(gvar.mean(spectrum) - _fit['fitted_vector'])
#         return fit_lines_in_spectrum(
#             params, inputparams, fit_tol, spectrum,
#             theta_orig, snr_guess=snr_guess,
#             max_iter=max_iter,
#             **kwargs_orig)
#     else:
#         return _fit


def fit_velocity_error_model(x, y, vel, vel_err, nm_laser,
                             calibration_laser_map,
                             pixel_size, binning=6):
    """Fit a model of the spectral calibration error based on a simplified
    optical model of the interferometer.

    :param x: Positions of the velocities along x axis

    :param y: Positions of the velocities along y axis

    :param vel: Measured velocity errors at (x, y)

    :param vel_err: Uncertainty on the measured velocity errors.

    :param nm_laser: Calibration laser wavelength in nm

    :param calibration_laser_map: Calibration laser map

    :param pixel_size: Pixel size in um

    param binning: (Optional) Binning during computation (process is faster with
      marginal precision loss) (default 6)
    """
    def model(p, wf, pixel_size, orig_fit_map, x, y):
        # 0: mirror_distance
        # 1: theta_cx
        # 2: theta_cy
        # 3: phi_x
        # 4: phi_y
        # 5: phi_r
        # 6: calib_laser_nm
        new_map = (orb.utils.image.simulate_calibration_laser_map(
            wf.shape[0], wf.shape[1], pixel_size,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6])
                + wf)
        dl_map = new_map - orig_fit_map
        dl_mod = list()
        for i in range(len(x)):
            dl_mod.append(dl_map[int(x[i]), int(y[i])])
        return np.array(dl_mod)

    def get_p(p_var, p_fix, p_ind):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        p_all = np.empty_like(p_ind, dtype=float)
        p_all[np.nonzero(p_ind == 0.)] = p_var
        p_all[np.nonzero(p_ind > 0.)] = p_fix
        return p_all

    def diff(p_var, p_fix, p_ind, wf, pixel_size, orig_fit_map, x, y, dl):
        p = get_p(p_var, p_fix, p_ind)
        dl_mod = model(p, wf, pixel_size, orig_fit_map, x, y)

        res = ((dl_mod - gvar.mean(dl))/gvar.sdev(dl)).astype(float)
        return res[~np.isnan(res)]

    def print_params(params):
        logging.info('    > New calibration laser map fit parameters:\n'
               + '    distance to mirror: {} cm\n'.format(
                   params[0] * 1e-4)
               + '    X angle from the optical axis to the center: {} degrees\n'.format(
                   np.fmod(float(params[1]),360))
               + '    Y angle from the optical axis to the center: {} degrees\n'.format(
                   np.fmod(float(params[2]),360))
               + '    Tip-tilt angle of the detector along X: {} degrees\n'.format(
                   np.fmod(float(params[3]),360))
               + '    Tip-tilt angle of the detector along Y: {} degrees\n'.format(
                   np.fmod(float(params[4]),360))
               + '    Rotation angle of the detector: {} degrees\n'.format(
                   np.fmod(float(params[5]),360))
               + '    Calibration laser wavelength: {} nm\n'.format(
                   params[6]))

    if (x.shape != y.shape or x.shape != vel.shape
        or vel.shape != vel_err.shape):
        raise TypeError('x, y, vel and vel_err must have the same shape')

    if x.ndim != 1: raise TypeError('x must have only one dimension')

    # create weights map
    w = 1./(vel_err)
    #w /= np.nanmax(w)
    #w[np.isnan(w)] = 1e-35
    x = x[~np.isnan(w)]
    y = y[~np.isnan(w)]
    vel = vel[~np.isnan(w)]
    vel_err= vel_err[~np.isnan(w)]
    w = w[~np.isnan(w)]

    vel[vel < np.nanpercentile(vel, 5)] = np.nan
    vel[vel > np.nanpercentile(vel, 95)] = np.nan
    x = x[~np.isnan(vel)]
    y = y[~np.isnan(vel)]
    w = w[~np.isnan(vel)]
    vel_err = vel_err[~np.isnan(vel)]
    vel = vel[~np.isnan(vel)]


    # transform velocity error in calibration error (v = dl/l with
    # l = 543.5 nm) (velocity error is the inverse of the velocity
    # measured)
    vel = -gvar.gvar(vel, vel_err)
    shift_map = orb.utils.spectrum.line_shift(
        vel, nm_laser)


    # compute a first estimation of the real calibration laser
    # wavelength
    new_nm_laser = nm_laser + np.nanmedian(gvar.mean(shift_map))
    logging.info('First laser wavelentgh calibration estimation: {} nm'.format(
        new_nm_laser))

    new_shift_map = shift_map - (new_nm_laser - nm_laser)

    # convert shift map to velocity map
    new_vel = orb.utils.spectrum.compute_radial_velocity(
        new_shift_map + new_nm_laser, new_nm_laser)

    # fit calibration map to get model + wavefront
    (orig_params,
     orig_fit_map,
     orig_model) = orb.utils.image.fit_calibration_laser_map(
        calibration_laser_map, new_nm_laser, pixel_size=pixel_size,
        return_model_fit=True)


    ######################
    ## orb.utils.io.write_fits('orig_fit_map.fits', orig_fit_map,
    ## overwrite=True)
    ## orb.utils.io.write_fits('orig_model.fits', orig_model, overwrite=True)
    ## orb.utils.io.write_fits('orig_params.fits', orig_params,
    ## overwrite=True)
    ## orig_fit_map = orb.utils.io.read_fits('orig_fit_map.fits')
    ## orig_model = orb.utils.io.read_fits('orig_model.fits')
    ## orig_params = orb.utils.io.read_fits('orig_params.fits')
    #################


    orig_fit_map_bin = orb.utils.image.nanbin_image(orig_fit_map, binning)
    orig_model_bin = orb.utils.image.nanbin_image(orig_model, binning)
    wf = orig_fit_map - orig_model
    wf_bin = orb.utils.image.nanbin_image(wf, binning)
    pixel_size_bin = pixel_size * float(binning)
    x_bin = x / float(binning)
    y_bin = y / float(binning)


    # calib laser map fit
    #p_var = orig_params[:-1]
    #p_fix = [new_nm_laser]
    #p_ind = np.array([0,0,0,0,0,0,1])
    p_var = orig_params[:-1]

    p_fix = []
    p_ind = np.array([0,0,0,0,0,0,0])
    fit = scipy.optimize.leastsq(diff,
                                 p_var,
                                 args=(p_fix, p_ind, wf_bin,
                                       pixel_size_bin,
                                       orig_fit_map_bin,
                                       x_bin, y_bin,
                                       new_shift_map),
                                 full_output=True)
    p = fit[0]
    print_params(p)

    # get fit stats
    model_shift_map = model(p, wf, pixel_size, orig_fit_map, x, y)
    model_vel = orb.utils.spectrum.compute_radial_velocity(
        new_nm_laser + model_shift_map, new_nm_laser,
        wavenumber=False)

    logging.info('fit residual std (in km/s):'.format(np.nanstd(
        model_vel - gvar.mean(vel))))

    logging.info('median error on the data (in km/s)'.format(np.nanmedian(
        gvar.sdev(vel))))

    # compute new calibration laser map
    model_calib_map = (orb.utils.image.simulate_calibration_laser_map(
        wf.shape[0], wf.shape[1], pixel_size,
        p[0], p[1], p[2], p[3], p[4], p[5], p[6])
                     + wf)

    # compute new velocity correction map
    final_shift_map = model_calib_map - orig_fit_map
    final_vel_map = orb.utils.spectrum.compute_radial_velocity(
        (new_nm_laser + final_shift_map), nm_laser,
        wavenumber=False)

    return model_calib_map, wf, final_vel_map, new_nm_laser


def get_timestamp():
    """Return a formatted timestamp.
    """
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
