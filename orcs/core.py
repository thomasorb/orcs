#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

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
ORCS Core library.

.. note:: ORCS is built over ORB so that ORB must be installed.
"""

import version
__version__ = version.__version__

# import Python libraries
import os
import logging
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import scipy.interpolate
import marshal
import time
import gvar
import warnings
import copy

# import ORB
try:
    import orb.core
    import orb.fit
    import orb.utils.astrometry
except Exception, e:
    print "ORB could not be found !"
    print e
    import sys
    sys.exit(2)

import utils
import rvcorrect

#################################################
#### CLASS HDFCube ##############################
#################################################

class HDFCube(orb.core.HDFCube):
    """Extension of :py:class:`orb.core.HDFCube`

    Core class which gives access to an HDF5 cube. The child class
    :py:class:`~orcs.process.SpectralCube` may be prefered in general
    for its broader functionality.

    .. seealso:: :py:class:`orb.core.HDFCube`
    """
    def __init__(self, cube_path, debug=False, **kwargs):
        """
        :param cube_path: Path to the HDF5 cube.

        :param kwargs: Kwargs are :meth:`orb.core.HDFCube` properties.
        """
        self.debug = debug
        self.logger = orb.core.Logger(debug=self.debug)
        FIT_TOL = 1e-10
        self.cube_path = cube_path
        instrument = None
        if 'SITELLE' in self.get_header()['INSTRUME']:
            instrument = 'sitelle'


        kwargs['instrument'] = instrument
        orb.core.HDFCube.__init__(self, cube_path, **kwargs)

        self.overwrite = True

        self.set_param('init_fwhm', float(self._get_config_parameter('INIT_FWHM')))
        self.set_param('fov', float(self._get_config_parameter('FIELD_OF_VIEW_1')))
        self.set_param('init_wcs_rotation', float(self._get_config_parameter('INIT_ANGLE')))


        self.fit_tol = FIT_TOL

        self.set_header(None)

    def _get_data_prefix(self):
        """Return data prefix"""
        return self._data_prefix

    def _get_reprojected_cube_path(self):
        """Return the path to the reprojected cube"""
        return self._data_path_hdr + 'reprojected_cube.fits'

    def _get_integrated_spectrum_path(self, region_name):
        """Return the path to an integrated spectrum

        :param region_name: Name of the region
        """
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integrated_spectrum_{}.fits".format(region_name))


    def _get_integrated_spectrum_header(self, region_name):
        """Return integrated spectrum header

        :param region_name: Region name
        """
        hdr = (self._get_basic_header('Integrated region {}'.format(region_name))
               + self._project_header
               + self._get_basic_spectrum_header(
                   self.params.base_axis.astype(float),
                   wavenumber=self.params.wavenumber))
        return hdr


    def _get_integrated_spectrum_fit_path(self, region_name):
        """Return the path to an integrated spectrum fit

        :param region_name: Name of the region
        """
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integrated_spectrum_fit_{}.fits".format(region_name))


    def _extract_spectrum_from_region(self, region,
                                      subtract_spectrum=None,
                                      median=False,
                                      mean_flux=False,
                                      silent=False,
                                      return_spec_nb=False,
                                      return_mean_theta=False,
                                      return_gvar=False,
                                      output_axis=None):
        """
        Extract the integrated spectrum from a region of the cube.

        All extraction of spectral data must use this core function
        because it makes sure that all the updated calibrations are
        taken into account.

        :param region: A list of the indices of the pixels integrated
          in the returned spectrum.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param median: (Optional) If True the integrated spectrum is computed
          from the median of the spectra multiplied by the number of
          pixels integrated. Else the integrated spectrum is the pure
          sum of the spectra. In both cases the flux of the spectrum
          is the total integrated flux (Default False).

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param return_spec_nb: (Optional) If True the number of
          spectra integrated is returned (default False).

        :param silent: (Optional) If True, nothing is printed (default
          False).

        :param return_mean_theta: (Optional) If True, the mean of the
          theta values covered by the region is returned (default False).

        :param return_gvar: (Optional) If True, returned spectrum will be a
          gvar. i.e. a data vector with it's uncetainty (default False).

        :param output_axis: (Optional) If not None, the spectrum is
          projected on the output axis. Else a scipy.UnivariateSpline
          object is returned (defautl None).

        :return: A scipy.UnivariateSpline object or a spectrum
          projected on the ouput_axis if it is not None.
        """
        def _interpolate_spectrum(spec, corr, wavenumber, step, order, base_axis):
            if wavenumber:
                corr_axis = orb.utils.spectrum.create_cm1_axis(
                    spec.shape[0], step, order, corr=corr)
                return orb.utils.vector.interpolate_axis(
                    spec, base_axis, 5, old_axis=corr_axis)
            else: raise NotImplementedError()


        def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
                                        median,
                                        wavenumber, base_axis, step, order,
                                        base_axis_corr):

            for icol in range(data_col.shape[0]):
                if mask_col[icol]:
                    corr = calib_coeff_col[icol]
                    if corr != base_axis_corr:
                        data_col[icol, :] = _interpolate_spectrum(
                            data_col[icol, :], corr, wavenumber, step, order, base_axis)
                else:
                    data_col[icol, :].fill(np.nan)

            if median:
                return (np.nanmedian(data_col, axis=0) * np.nansum(mask_col),
                        np.nansum(mask_col))
            else:
                return (np.nansum(data_col, axis=0),
                        np.nansum(mask_col))

        if median:
            warnings.warn('Median integration')


        calibration_coeff_map = self.get_calibration_coeff_map()

        calibration_coeff_center = calibration_coeff_map[
            calibration_coeff_map.shape[0]/2,
            calibration_coeff_map.shape[1]/2]

        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1
        if not silent:
            logging.info('Number of integrated pixels: {}'.format(np.sum(mask)))

        if np.sum(mask) == 0: raise StandardError('A region must contain at least one valid pixel')

        elif np.sum(mask) == 1:
            ii = region[0][0] ; ij = region[1][0]
            spectrum = _interpolate_spectrum(
                self.get_data(ii, ii+1, ij, ij+1, 0, self.dimz, silent=silent),
                calibration_coeff_map[ii, ij],
                self.params.wavenumber, self.params.step, self.params.order,
                self.params.base_axis)
            counts = 1

        else:
            spectrum = np.zeros(self.dimz, dtype=float)
            counts = 0

            # get range to check if a quadrants extraction is necessary
            mask_x_proj = np.nanmax(mask, axis=1).astype(float)
            mask_x_proj[np.nonzero(mask_x_proj == 0)] = np.nan
            mask_x_proj *= np.arange(self.dimx)
            x_min = int(np.nanmin(mask_x_proj))
            x_max = int(np.nanmax(mask_x_proj)) + 1

            mask_y_proj = np.nanmax(mask, axis=0).astype(float)
            mask_y_proj[np.nonzero(mask_y_proj == 0)] = np.nan
            mask_y_proj *= np.arange(self.dimy)
            y_min = int(np.nanmin(mask_y_proj))
            y_max = int(np.nanmax(mask_y_proj)) + 1

            if (x_max - x_min < self.dimx / float(self.config.DIV_NB)
                and y_max - y_min < self.dimy / float(self.config.DIV_NB)):
                quadrant_extraction = False
                QUAD_NB = 1
                DIV_NB = 1
            else:
                quadrant_extraction = True
                QUAD_NB = self.config.QUAD_NB
                DIV_NB = self.config.DIV_NB


            for iquad in range(0, QUAD_NB):

                if quadrant_extraction:
                    # x_min, x_max, y_min, y_max are now used for quadrants boundaries
                    x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)

                iquad_data = self.get_data(x_min, x_max, y_min, y_max,
                                           0, self.dimz, silent=silent)

                # multi-processing server init
                job_server, ncpus = self._init_pp_server(silent=silent)
                if not silent: progress = orb.core.ProgressBar(x_max - x_min)
                for ii in range(0, x_max - x_min, ncpus):
                    # no more jobs than columns
                    if (ii + ncpus >= x_max - x_min):
                        ncpus = x_max - x_min - ii

                    # jobs creation
                    jobs = [(ijob, job_server.submit(
                        _extract_spectrum_in_column,
                        args=(iquad_data[ii+ijob,:,:],
                              calibration_coeff_map[x_min + ii + ijob,
                                                    y_min:y_max],
                              mask[x_min + ii + ijob, y_min:y_max],
                              median, self.params.wavenumber,
                              self.params.base_axis, self.params.step,
                              self.params.order, self.params.axis_corr),
                        modules=("import logging",
                                 'import numpy as np',
                                 'import orb.utils.spectrum',
                                 'import orb.utils.vector'),
                        depfuncs=(_interpolate_spectrum,)))
                            for ijob in range(ncpus)]

                    for ijob, job in jobs:
                        spec_to_add, spec_nb = job()
                        if not np.all(np.isnan(spec_to_add)):
                            spectrum += spec_to_add
                            counts += spec_nb

                    if not silent:
                        progress.update(ii, info="ext column : {}/{}".format(
                            ii, int(self.dimx/float(DIV_NB))))
                self._close_pp_server(job_server)
                if not silent: progress.end()


        # add uncertainty on the spectrum
        if return_gvar:
            flux_uncertainty = self.get_flux_uncertainty()

            if flux_uncertainty is not None:
                uncertainty = np.nansum(flux_uncertainty[np.nonzero(mask)])
                logging.debug('computed mean flux uncertainty: {}'.format(uncertainty))
                spectrum = gvar.gvar(spectrum, np.ones_like(spectrum) * uncertainty)


        if subtract_spectrum is not None:
            spectrum -= subtract_spectrum * counts

        if mean_flux:
            spectrum /= counts

        returns = list()
        if output_axis is not None and np.all(output_axis == self.params.base_axis):
            spectrum[np.isnan(gvar.mean(spectrum))] = 0. # remove nans
            returns.append(spectrum)

        else:
            nonans = ~np.isnan(gvar.mean(spectrum))
            spectrum_function = scipy.interpolate.UnivariateSpline(
                self.params.base_axis[nonans], gvar.mean(spectrum)[nonans],
                s=0, k=1, ext=1)
            if return_gvar:
                spectrum_function_sdev = scipy.interpolate.UnivariateSpline(
                    self.params.base_axis[nonans], gvar.sdev(spectrum)[nonans],
                    s=0, k=1, ext=1)
                raise Exception('now a tuple is returned with both functions for mean and sdev, this will raise an error somewhere and must be checked before')
                spectrum_function = (spectrum_function, spectrum_function_sdev)

            if output_axis is None:
                returns.append(spectrum_function(gvar.mean(output_axis)))
            else:
                returns.append(spectrum_function)

        if return_spec_nb:
            returns.append(counts)
        if return_mean_theta:
            theta_map = self.get_theta_map()
            mean_theta = np.nanmean(theta_map[np.nonzero(mask)])
            logging.debug('computed mean theta: {}'.format(mean_theta))
            returns.append(mean_theta)

        return returns


    def _fit_lines_in_region(self, region, subtract_spectrum=None,
                             binning=1, snr_guess=None):
        """
        Raw function that fit lines in a given region of the cube.


        All the pixels in the defined region are fitted one by one
        and a set of maps containing the fitted paramaters are
        written. Note that the pixels can be binned.


        .. note:: Need the InputParams class to be defined before call
          (see :py:meth:`~orcs.core.HDFCube._prepare_input_params`).

        .. note:: The fit will always use the Bayesian algorithm.

        :param region: Region to fit. Multiple regions can be used to
          define the fitted region. They do not need to be contiguous.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param binning: (Optional) Binning. The fitted pixels can be
          binned.

        :param snr_guess: Guess on the SNR of the spectrum. Can only
          be None or 'auto'. Set it to 'auto' to make a Bayesian
          fit. In this case two fits are made - one with a predefined
          SNR and the other with the SNR deduced from the first
          fit. If None a classical fit is made.

        .. note:: Maps of the parameters of the fit can be found in
          the directory created by ORCS:
          ``OBJECT_NAME_FILTER.ORCS/MAPS/``.

          Each line has 5 parameters (which gives 5 maps): height,
          amplitude, velocity, fwhm, sigma. Height and amplitude are
          given in ergs/cm^2/s/A. Velocity and broadening are given in
          km/s. FWHM is given in cm^-1.

          The flux map is also computed (from fwhm, amplitude and
          sigma parameters) and given in ergs/cm^2/s.

          Each fitted parameter is associated an uncertainty (``*_err``
          maps) given in the same unit.

        """
        def fit_lines_in_pixel(spectrum, params, inputparams, fit_tol,
                               init_velocity_map_ij, init_sigma_map_ij,
                               theta_map_ij, snr_guess, sky_vel_ij, flux_sdev_ij,
                               debug):

            stime = time.time()
            if debug:
                logging.getLogger().setLevel(logging.DEBUG)
            else:
                warnings.simplefilter('ignore', RuntimeWarning)
            # check init velocity (already computed from
            # binned maps). If a guess exists, velocity range
            # is set to None
            if not np.isnan(init_velocity_map_ij):
                shift_guess_sdev = gvar.sdev(
                    inputparams.allparams['pos_cov'])
                shift_guess = gvar.gvar(
                    np.ones_like(shift_guess_sdev) * init_velocity_map_ij,
                    shift_guess_sdev)
            else:
                shift_guess = None

            # check init sigma (already computed from binned
            # maps).
            if not np.isnan(init_sigma_map_ij):
                sigma_guess_sdev = gvar.sdev(
                    inputparams.allparams['sigma_cov'])
                sigma_guess = gvar.gvar(
                    np.ones_like(sigma_guess_sdev) * init_sigma_map_ij,
                    sigma_guess_sdev)
            else:
                sigma_guess = None

            # correct spectrum for nans
            spectrum[np.isnan(spectrum)] = 0.

            # add flux uncertainty to the spectrum
            spectrum = gvar.gvar(spectrum, np.ones_like(spectrum) * flux_sdev_ij)

            try:
                ifit = orcs.utils.fit_lines_in_spectrum(
                    params, inputparams, fit_tol, spectrum, theta_map_ij,
                    snr_guess=snr_guess, sigma_guess=sigma_guess,
                    pos_cov=shift_guess)
            except Exception, e:
                logging.debug('Exception occured during fit: {}'.format(e))
                ifit = []

            if ifit != []:
                logging.debug('pure fit time: {} s'.format(ifit['fit_time']))
                logging.debug('fit function time: {} s'.format(time.time() - stime))
                logging.debug('velocity: {}'.format(ifit['velocity_gvar'] + sky_vel_ij))
                logging.debug('broadening: {}'.format(ifit['broadening_gvar']))

                return {
                    'height': ifit['lines_params'][:,0],
                    'amplitude': ifit['lines_params'][:,1],
                    'fwhm': ifit['lines_params'][:,3],
                    'velocity': ifit['velocity'] + sky_vel_ij,
                    'sigma': ifit['broadening'],
                    'flux': ifit['flux'],
                    'height-err': ifit['lines_params_err'][:,0],
                    'amplitude-err': ifit['lines_params_err'][:,1],
                    'fwhm-err': ifit['lines_params_err'][:,3],
                    'velocity-err': ifit['velocity_err'],
                    'sigma-err': ifit['broadening_err'],
                    'flux-err': ifit['flux_err'],
                    'chi2': ifit['chi2'],
                    'rchi2': ifit['rchi2'],
                    'logGBF': ifit['logGBF'],
                    'ks_pvalue': ifit['ks_pvalue']}

            else:
                return {
                    'height': None,
                    'amplitude': None,
                    'fwhm': None,
                    'velocity': None,
                    'sigma': None,
                    'flux': None,
                    'height-err': None,
                    'amplitude-err': None,
                    'fwhm-err': None,
                    'velocity-err': None,
                    'sigma-err': None,
                    'flux-err': None,
                    'chi2': None,
                    'rchi2': None,
                    'logGBF': None,
                    'ks_pvalue': None}

        if snr_guess not in ('auto', None):
            raise ValueError("snr_guess must be 'auto' or None")

        ## check if input params is instanciated
        if not hasattr(self, 'inputparams'):
            raise StandardError('Input params not defined')

        ## init LineMaps object
        linemaps = LineMaps(
            self.dimx, self.dimy, gvar.mean(
                self.inputparams.allparams['pos_guess']),
            self.params.wavenumber,
            binning, self.config.DIV_NB,
            instrument=self.instrument,
            project_header=self._project_header,
            wcs_header=self.wcs_header,
            data_prefix=self._data_prefix,
            ncpus=self.ncpus)

        # compute max uncertainty on velocity and sigma to use it as
        # an initial guess.
        max_vel_err = (orb.constants.LIGHT_VEL_KMS / self.params.resolution) / 4.
        max_sig_err = max_vel_err / 2.

        # load velocity maps of binned maps
        init_velocity_map = linemaps.get_map('velocity')
        init_velocity_map_err = linemaps.get_map('velocity-err')
        # load sigma maps of binned maps
        init_sigma_map = linemaps.get_map('sigma')
        init_sigma_map_err = linemaps.get_map('sigma-err')

        # create mean velocity map
        # mean map is weighted by the square of the error on the parameter
        init_velocity_map[
            np.nonzero(np.abs(init_velocity_map_err) > max_vel_err)] = np.nan
        vel_map_w = (1. / (init_velocity_map_err**2.))
        init_velocity_map *= vel_map_w
        init_velocity_map = np.nansum(init_velocity_map, axis=2)
        init_velocity_map /= np.nansum(vel_map_w, axis=2)

        # create mean sigma map
        # mean map is weighted by the square of the error on the parameter
        init_sigma_map[
            np.nonzero(np.abs(init_sigma_map_err) > max_sig_err)] = np.nan
        sig_map_w = (1. / (init_sigma_map_err**2.))
        init_sigma_map *= sig_map_w
        init_sigma_map = np.nansum(init_sigma_map, axis=2)
        init_sigma_map /= np.nansum(sig_map_w, axis=2)

        # check subtract spectrum
        if np.all(subtract_spectrum == 0.): subtract_spectrum = None

        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[region] = True

        theta_map = orb.utils.image.nanbin_image(self.get_theta_map(), binning)

        mask = np.zeros((self.dimx, self.dimy), dtype=float)
        mask[region] = 1

        mask_bin = orb.utils.image.nanbin_image(mask, binning)

        total_fit_nb = np.nansum(mask_bin)
        progress = orb.core.ProgressBar(total_fit_nb)
        logging.info('Number of spectra to fit: {}'.format(int(total_fit_nb)))

        if self.get_sky_velocity_map() is not None:
            sky_velocity_map = orb.utils.image.nanbin_image(
                self.get_sky_velocity_map(), binning)
        else:
            sky_velocity_map = orb.utils.image.nanbin_image(
                np.zeros((self.dimx, self.dimy), dtype=float), binning)

        flux_uncertainty = self.get_flux_uncertainty()
        if flux_uncertainty is not None:
            flux_uncertainty = orb.utils.image.nanbin_image(flux_uncertainty, binning)
        else:
            flux_uncertainty = orb.utils.image.nanbin_image(
                np.ones((self.dimx, self.dimy), dtype=float), binning) * binning**2.

        cjs = CubeJobServer(self)
        out = cjs.process_by_pixel(fit_lines_in_pixel,
                             args=[self.params.convert(), self.inputparams.convert(), self.fit_tol,
                                   init_velocity_map, init_sigma_map,
                                   theta_map, snr_guess, sky_velocity_map,
                                   flux_uncertainty, self.debug],
                             modules=['numpy as np', 'gvar', 'orcs.utils',
                                      'logging', 'warnings', 'time'],
                             mask=mask,
                             binning=binning)

        for key in out:
            linemaps.set_map(key, out[key],
                             x_range=[0, self.dimx],
                             y_range=[0, self.dimy])


        linemaps.write_maps()

    def _fit_integrated_spectra(self, regions_file_path,
                               subtract=None,
                               plot=True,
                               verbose=True):
        """
        Fit integrated spectra and their emission lines parameters.

        .. note:: Raw function which needs self.inputparams to be
          defined before with
          `:py:meth:~HDFCube._prepare_input_params`.

        :param regions_file_path: Path to a ds9 reg file giving the
          positions of the regions. Each region is considered as a
          different region.

        :param subtract: Spectrum to subtract (must be a spline)

        :param plot: (Optional) If True, plot each intergrated spectrum along
          with the its fit (default True).

        :param verbose: (Optional) If True print the fit results
          (default True).
        """
        def _fit_lines(spectrum, theta_orig, params, inputparams, fit_tol):
            return utils.fit_lines_in_spectrum(
                params, inputparams, fit_tol,
                spectrum, theta_orig, snr_guess=None).convert()


        if verbose:
            logging.info("Extracting integrated spectra")

        calibration_coeff_map = self.get_calibration_coeff_map()
        calibration_laser_map = self.get_calibration_laser_map()

        # Create parameters file
        paramsfile = list()

        # extract and fit regions
        integ_spectra = list()

        regions = self.get_mask_from_ds9_region_file(
            regions_file_path, integrate=False)


        if not (self.params.wavelength_calibration):
            raise Exception('Not implemented')
        else:
            axis = self.params.base_axis.astype(float)

        if not hasattr(self, 'inputparams'):
            raise StandardError('Input params not defined')


        cjs = CubeJobServer(self)
        all_fit = cjs.process_by_region(
            _fit_lines, regions, subtract, axis,
            args=(self.params.convert(), self.inputparams.convert(), self.fit_tol),
            modules=('import logging',
                     'import orcs.utils as utils'))

        lines = gvar.mean(self.inputparams.allparams.pos_guess)

        # process results
        for iregion in range(len(regions)):
            ifit, ispectrum = all_fit[iregion]

            if ifit != []:

                all_fit_results = list()
                logging.info('Velocity of the first line (km/s): {}'.format(ifit['velocity_gvar'][0]))
                line_names = list()
                for iline in range(np.size(lines)):
                    if self.params.wavenumber:
                        line_name = orb.core.Lines().round_nm2ang(
                            orb.utils.spectrum.cm12nm(
                                lines[iline]))
                    else:
                        line_name = orb.core.Lines().round_nm2ang(
                            lines[iline])

                    fit_params = ifit['lines_params']
                    err_params = ifit['lines_params_err']

                    fit_results = {
                        'reg_index': iregion,
                        'line_name': line_name,
                        'h': fit_params[iline, 0],
                        'a': fit_params[iline, 1],
                        'x': fit_params[iline, 2],
                        'v': ifit['velocity'][iline],
                        'fwhm': fit_params[iline, 3],
                        'sigma': fit_params[iline, 4],
                        'broadening': ifit['broadening'][iline],
                        'flux': ifit['flux'][iline],
                        'h_err': err_params[iline, 0],
                        'a_err': err_params[iline, 1],
                        'x_err': err_params[iline, 2],
                        'v_err': ifit['velocity_err'][iline],
                        'broadening_err': ifit['broadening_err'][iline],
                        'fwhm_err': err_params[iline, 3],
                        'sigma_err': err_params[iline, 4],
                        'flux_err': ifit['flux_err'][iline]}
                    all_fit_results.append(fit_results)


                fitted_vector = ifit['fitted_vector']
                fitted_models = ifit['fitted_models']
            else:
                all_fit_results = [dict() for _ in range(len(lines))]
                fitted_vector = np.zeros_like(ispectrum)
                fitted_models = list()

            if self.params.wavenumber:
                linesmodel = 'Cm1LinesModel'
            else:
                raise NotImplementedError()
                linesmodel = 'NmLinesModel'

            spectrum_header = (
                self._get_integrated_spectrum_header(
                    iregion))

            self.write_fits(
                self._get_integrated_spectrum_path(
                    iregion),
                ispectrum, fits_header=spectrum_header,
                overwrite=True, silent=True)

            self.write_fits(
                self._get_integrated_spectrum_fit_path(
                    iregion),
                fitted_vector,
                fits_header=spectrum_header,
                overwrite=True, silent=True)

            for fit_results in all_fit_results:
                paramsfile.append(fit_results)
                if verbose:
                    logging.info(
                        '================ Line: {} ============='.format(
                            fit_results['line_name']))
                    for ikey in fit_results:
                        logging.info('{}: {}'.format(
                            ikey, fit_results[ikey]))

            if plot and ifit != []:

                import pylab as pl
                ax1 = pl.subplot(211)
                ax1.plot(axis, ispectrum, c= '0.3',
                         ls='--', lw=1.5,
                         label='orig spectrum')

                for imod in range(len(fitted_models[linesmodel])):
                    ax1.plot(
                        axis,
                        fitted_models[linesmodel][imod]
                        + fitted_models['ContinuumModel'],
                        c= '0.5', ls='-', lw=1.5)

                ax1.plot(axis, fitted_vector, c= '0.',
                         ls='-', lw=1.5,
                         label='fit')
                ax1.grid()
                ax1.legend()
                ax2 = pl.subplot(212, sharex=ax1)
                ax2.plot(axis, ispectrum - fitted_vector, c= 'red',
                         ls='-', lw=1.5, label='residual')
                ax2.grid()
                ax2.legend()
                pl.show()

            integ_spectra.append(ispectrum)

        return paramsfile

    def _fit_lines_in_spectrum(self, spectrum, theta_orig, snr_guess=None, **kwargs):
        """Raw function for spectrum fitting.

        .. note:: Need the InputParams class to be defined before call
        (see :py:meth:`~orcs.core.HDFCube._prepare_input_params`).

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

        if not hasattr(self, 'inputparams'):
            raise StandardError('Input params not defined')

        return utils.fit_lines_in_spectrum(
            self.params, self.inputparams, self.fit_tol,
            spectrum, theta_orig, snr_guess=snr_guess, **kwargs)

    def _prepare_input_params(self, lines, nofilter=False, **kwargs):
        """prepare the InputParams instance for a fitting procedure.

        :param lines: Emission lines to fit (must be in cm-1 if the
          cube is in wavenumber. must be in nm otherwise).

        :param nofilter: (Optional) If True, Filter model is not added
          and the fit is made with a single range set to the filter
          bandpass.

        :param kwargs: Keyword arguments of the function
          :py:meth:`orb.fit._prepare_input_params`.
        """

        if nofilter:
            filter_file_path = None
            if 'signal_range' in kwargs:
                signal_range = kwargs['signal_range']
                del kwargs['signal_range']
            else:
                signal_range = self.get_filter_range()

        else:
            filter_file_path = self.params.filter_file_path
            if 'signal_range' in kwargs:
                signal_range = kwargs['signal_range']
                del kwargs['signal_range']
            else:
                signal_range = None

        self.inputparams = orb.fit._prepare_input_params(
            self.params.step_nb,
            lines,
            self.params.step,
            self.params.order,
            self.params.nm_laser,
            self.params.theta_proj,
            self.params.zpd_index,
            ## warning, theta_orig set to theta_proj by default.
            ## Means that the real theta_orig must be defined in the
            ## fitting function _fit_lines_in_spectrum
            theta_orig=self.params.theta_proj,
            wavenumber=self.params.wavenumber,
            filter_file_path=filter_file_path,
            apodization=self.params.apodization,
            signal_range=signal_range,
            **kwargs)


    def get_calibration_coeff_map(self):
        """Return the calibration coeff map based on the calibration
        laser map and the laser wavelength.
        """
        if hasattr(self, 'calibration_coeff_map'):
            return self.calibration_coeff_map
        else:
            self.calibration_coeff_map = self.get_calibration_laser_map() / self.params.nm_laser
        return self.calibration_coeff_map

    def get_calibration_laser_map(self):
        """Return the calibration laser map of the cube"""
        if hasattr(self, 'calibration_laser_map'):
            return self.calibration_laser_map

        calib_map = self.get_calibration_laser_map_orig()
        if calib_map is None:
            raise StandardError('No calibration laser map given. Please redo the last step of the data reduction')

        if self.params.wavelength_calibration:
            calib_map = (np.ones((self.dimx, self.dimy), dtype=float)
                    * self.params.nm_laser * self.params.axis_corr)

        elif (calib_map.shape[0] != self.dimx):
            calib_map = orb.utils.image.interpolate_map(
                calib_map, self.dimx, self.dimy)

        # calibration correction
        if self.get_sky_velocity_map() is not None:
            ratio = 1 + (self.get_sky_velocity_map() / orb.constants.LIGHT_VEL_KMS)
            calib_map /= ratio

        self.calibration_laser_map = calib_map
        self.reset_calibration_coeff_map()
        return self.calibration_laser_map

    def get_sky_velocity_map(self):
        if hasattr(self, 'sky_velocity_map'):
            return self.sky_velocity_map
        else: return None

    def get_fwhm_map(self):
        """Return the theoretical FWHM map in cm-1 based only on the angle
        and the theoretical attained resolution."""
        return orb.utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index, self.params.step, self.params.order,
            self.params.apodization, self.get_calibration_coeff_map_orig(),
            wavenumber=self.params.wavenumber)

    def get_theta_map(self):
        """Return the incident angle map (in degree)"""
        return orb.utils.spectrum.corr2theta(self.get_calibration_coeff_map_orig())

    def reset_calibration_laser_map(self):
        """Reset the compute calibration laser map (and also the
        calibration coeff map). Must be called when the wavelength
        calibration has changed

        ..seealso :: :py:meth:`~HDFCube.correct_wavelength`
        """
        if hasattr(self, 'calibration_laser_map'):
            del self.calibration_laser_map
        self.reset_calibration_coeff_map()

    def reset_calibration_coeff_map(self):
        """Reset the computed calibration coeff map alone"""
        if hasattr(self, 'calibration_coeff_map'):
            del self.calibration_coeff_map

    def get_calibration_laser_map_orig(self):
        """Return the original calibration laser map (not the version
        computed by :py:meth:`~HDFCube.get_calibration_laser_map`)"""
        return orb.core.HDFCube.get_calibration_laser_map(self)

    def get_calibration_coeff_map_orig(self):
        """Return the original calibration coeff map (not the version
        computed by :py:meth:`~HDFCube.get_calibration_coeff_map`)"""
        return self.get_calibration_laser_map_orig() / self.params.nm_laser


    def get_filter_range(self):
        """Return the range of the filter in the unit of the spectral
        cube as a tuple (min, max)"""
        _range = orb.utils.filters.get_filter_bandpass(
            self.params.filter_file_path)
        if self.params.wavenumber:
            _range = orb.utils.spectrum.nm2cm1(_range)
        return [min(_range), max(_range)]


    def get_sky_lines(self):
        """Return the wavenumber/wavelength of the sky lines in the
        filter range"""
        _delta_nm = self.params.axis_step
        if self.params.wavenumber:
            _delta_nm = orb.utils.spectrum.fwhm_cm12nm(
                self.params.axis_step,
                (self.params.axis_min + self.params.axis_max) / 2.)

        _nm_min, _nm_max = self.get_filter_range()

        # we add 5% to the computed size of the filter
        _nm_range = _nm_max - _nm_min
        _nm_min -= _nm_range * 0.05
        _nm_max += _nm_range * 0.05

        if self.params.wavenumber:
            _nm_max, _nm_min = orb.utils.spectrum.cm12nm([_nm_min, _nm_max])

        _lines_nm = orb.core.Lines().get_sky_lines(
            _nm_min, _nm_max, _delta_nm)

        if self.params.wavenumber:
            return orb.utils.spectrum.nm2cm1(_lines_nm)
        else:
            return _line_nm

    def extract_spectrum_bin(self, x, y, b, **kwargs):
        """Extract a spectrum integrated over a binned region.

        :param x: X position of the bottom-left pixel

        :param y: Y position of the bottom-left pixel

        :param b: Binning. If 1, only the central pixel is extracted

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if b < 1: raise StandardError('Binning must be at least 1')

        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[int(x):int(x+b), int(y):int(y+b)] = True
        region = np.nonzero(mask)

        return self.extract_integrated_spectrum(region, **kwargs)

    def extract_spectrum(self, x, y, r, **kwargs):
        """Extract a spectrum integrated over a circular region of a
        given radius.

        :param x: X position of the center

        :param y: Y position of the center

        :param r: Radius. If 0, only the central pixel is extracted.

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if r < 0: r = 0.001
        X, Y = np.mgrid[0:self.dimx, 0:self.dimy]
        R = np.sqrt(((X-x)**2 + (Y-y)**2))
        region = np.nonzero(R <= r)

        return self.extract_integrated_spectrum(region, **kwargs)


    def extract_integrated_spectrum(self, region, **kwargs):
        """Extract a spectrum integrated over a given region (can be a
        list of pixels as returned by the function
        :py:meth:`numpy.nonzero` or a ds9 region file).

        :param region: Region to integrate (can be a list of pixel
          coordinates as returned by the function
          :py:meth:`numpy.nonzero` or the path to a ds9 region
          file). If it is a ds9 region file, multiple regions can be
          defined and all will be integrated into one spectrum.
        """
        region = self.get_mask_from_ds9_region_file(region)

        returns = list()
        returns.append(self.params.base_axis.astype(float))
        returns += list(self._extract_spectrum_from_region(
            region, output_axis=self.params.base_axis.astype(float), **kwargs))
        return returns

    def fit_lines_in_spectrum_bin(self, x, y, b, lines, nofilter=False,
                                  subtract_spectrum=None,
                                  snr_guess=None,
                                  mean_flux=False,
                                  **kwargs):
        """Fit lines of a spectrum extracted from a squared region of a
        given size.

        :param x: X position of the bottom-left pixel

        :param y: Y position of the bottom-left pixel

        :param b: Binning. If 0, only the central pixel is extracted.

        :param lines: Emission lines to fit (must be in cm-1 if the
          cube is in wavenumber. must be in nm otherwise).

        :param nofilter: (Optional) If True, Filter model is not added
          and the fit is made with a single range set to the filter
          bandpass.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param snr_guess: Guess on the SNR of the spectrum. Necessary
          to make a Bayesian fit (If unknown you can set it to 'auto'
          to try an automatic mode, two fits are made - one with a
          predefined SNR and the other with the SNR deduced from the
          first fit). If None a classical fit is made.

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param kwargs: Keyword arguments of the function
          :py:meth:`orb.fit.fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`orb.fit.fit_lines_in_spectrum`)
        """

        axis, spectrum, theta_orig = self.extract_spectrum_bin(
            x, y, b, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux,
            return_mean_theta=True, return_gvar=True)

        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)

        fit_res = self._fit_lines_in_spectrum(
            spectrum, theta_orig, snr_guess=snr_guess)

        return axis, gvar.mean(spectrum), fit_res

    def fit_lines_in_spectrum(self, x, y, r, lines, nofilter=False,
                              snr_guess=None, subtract_spectrum=None,
                              mean_flux=False, **kwargs):
        """Fit lines of a spectrum extracted from a circular region of a
        given radius.

        :param x: X position of the center

        :param y: Y position of the center

        :param r: Radius. If 0, only the central pixel is extracted.

        :param lines: Emission lines to fit (must be in cm-1 if the
          cube is in wavenumber. must be in nm otherwise).

        :param nofilter: (Optional) If True, Filter model is not added
          and the fit is made with a single range set to the filter
          bandpass.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param snr_guess: Guess on the SNR of the spectrum. Necessary
          to make a Bayesian fit (If unknown you can set it to 'auto'
          to try an automatic mode, two fits are made - one with a
          predefined SNR and the other with the SNR deduced from the
          first fit). If None a classical fit is made.

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`~HDFCube._fit_lines_in_spectrum`)
        """
        axis, spectrum, theta_orig = self.extract_spectrum(
            x, y, r, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux,
            return_mean_theta=True, return_gvar=True)

        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)

        fit_res = self._fit_lines_in_spectrum(
            spectrum, theta_orig, snr_guess=snr_guess)

        return axis, gvar.mean(spectrum), fit_res

    def fit_lines_in_integrated_region(self, region, lines, nofilter=False,
                                       snr_guess=None, subtract_spectrum=None,
                                       mean_flux=False,
                                       **kwargs):
        """Fit lines of a spectrum integrated over a given region (can
        be a list of pixels as returned by the function
        :py:meth:`numpy.nonzero` or a ds9 region file).

        :param region: Region to integrate (can be a list of pixel
          coordinates as returned by the function
          :py:meth:`numpy.nonzero` or the path to a ds9 region
          file). If it is a ds9 region file, multiple regions can be
          defined and all will be integrated into one spectrum.

        :param lines: Emission lines to fit (must be in cm-1 if the
          cube is in wavenumber. must be in nm otherwise).

        :param nofilter: (Optional) If True, Filter model is not added
          and the fit is made with a single range set to the filter
          bandpass.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param snr_guess: Guess on the SNR of the spectrum. Necessary
          to make a Bayesian fit (If unknown you can set it to 'auto'
          to try an automatic mode, two fits are made - one with a
          predefined SNR and the other with the SNR deduced from the
          first fit). If None a classical fit is made.

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`~HDFCube._fit_lines_in_spectrum`)

        """
        axis, spectrum, theta_orig = self.extract_integrated_spectrum(
            region, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux,
            return_mean_theta=True, return_gvar=True)

        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)

        fit_res = self._fit_lines_in_spectrum(
            spectrum, theta_orig, snr_guess=snr_guess)

        return axis, gvar.mean(spectrum), fit_res

    def fit_lines_in_region(self, region, lines, binning=1, nofilter=False,
                            subtract_spectrum=None, snr_guess=None, **kwargs):
        """Fit lines in a given region of the cube. All the pixels in
        the defined region are fitted one by one and a set of maps
        containing the fitted paramaters are written. Note that the
        pixels can be binned.

        :param lines: Emission lines to fit (must be in cm-1 if the
          cube is in wavenumber. must be in nm otherwise).

        :param region: Region to fit. Multiple regions can be used to
          define the fitted region. They do not need to be contiguous.

        :param nofilter: (Optional) If True, Filter model is not added
          and the fit is made with a single range set to the filter
          bandpass.

        :param subtract_spectrum: (Optional) Remove the given spectrum
          from the extracted spectrum before fitting
          parameters. Useful to remove sky spectrum. Both spectra must
          have the same size.

        :param snr_guess: Guess on the SNR of the spectrum. Can only
          be None or 'auto'. Set it to 'auto' to make a Bayesian
          fit. In this case two fits are made - one with a predefined
          SNR and the other with the SNR deduced from the first
          fit. If None a classical fit is made.

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.
        """
        region = self.get_mask_from_ds9_region_file(region)
        self._prepare_input_params(
            lines, nofilter=nofilter, **kwargs)
        self._fit_lines_in_region(
            region, subtract_spectrum=subtract_spectrum,
            binning=binning, snr_guess=snr_guess)

    def get_mask_from_ds9_region_file(self, region, integrate=True):
        """Return a mask from a ds9 region file.

        :param region: Path to a ds9 region file.

        :param integrate: (Optional) If True, all pixels are integrated
          into one mask, else a list of region masks is returned (default
          True)
        """
        if isinstance(region, str):
            return orb.utils.misc.get_mask_from_ds9_region_file(
                region,
                [0, self.dimx],
                [0, self.dimy],
                header=self.header,
                integrate=integrate)
        else: return region

    def correct_wavelength(self, sky_map_path):
        """Correct the wavelength of the cube based on the velocity of
        the sky lines computed with
        :py:meth:`~orcs.process.SpectralCube.map_sky_velocity`

        :param sky_map_path: Path to the sky velocity map.

        .. warning:: the sky velocity map returned by the function
          SpectralCube.map_sky_velocity is inversed (a velocity of 80
          km/s is indicated as -80 km/s). It is thus more a correction
          map that must be directly added to the computed velocity to
          obtain a corrected velocity. As this map can be passed as
          is, it means that the given sky velocity map must be a
          correction map.
        """
        sky_map = self.read_fits(sky_map_path)
        if sky_map.shape != (self.dimx, self.dimy):
            raise StandardError('Given sky map does not have the right shape')

        self.sky_velocity_map = sky_map
        self.reset_calibration_laser_map()

    def get_amp_ratio_from_flux_ratio(self, line0, line1, flux_ratio):
        """Return the amplitude ratio (amp(line0) / amp(line1)) to define from the flux ratio
        (at constant fwhm and broadening).

        :param line0: Wavenumber of the line 0 (in cm-1).

        :param line1: Wavenumber of the line 1 (in cm-1).

        :param flux_ratio: Flux ratio: flux(line0) / flux(line1).
        """
        if isinstance(line0, str):
            line0 = orb.core.Lines().get_line_cm1(line0)
        if isinstance(line1, str):
            line1 = orb.core.Lines().get_line_cm1(line1)
        return line0**2 / line1**2 * flux_ratio

    def set_dxdymaps(self, dxmap_path, dymap_path):
        """Set micro-shift maps returned by the astrometrical
        calibration method.

        :param dxmap_path: Path to the dxmap.

        :param dymap_path: Path to the dymap.
        """
        dxmap = self.read_fits(dxmap_path)
        dymap = self.read_fits(dymap_path)
        if dxmap.shape == (self.dimx, self.dimy):
            if dymap.shape == (self.dimx, self.dimy):
                self.dxmap = dxmap
                self.dymap = dymap
            else:
                dymap = orb.utils.image.interpolate_map(
                    dymap, self.dimx, self.dimy)
                warnings.warn('dymap reshaped from {} to ({}, {})'.
                                 format(dymap.shape, self.dimx, self.dimy))

        else:
            dxmap = orb.utils.image.interpolate_map(
                dxmap, self.dimx, self.dimy)
            warnings.warn('dxmap reshaped from {} to ({}, {})'.
                             format(dxmap.shape, self.dimx, self.dimy))

    def set_wcs(self, wcs_path):
        """Reset WCS of the cube.

        :param wcs_path: Path to a FITS image containing the new WCS.
        """
        hdr = self.get_header()

        hdr.update(pywcs.WCS(self.read_fits(wcs_path, return_hdu_only=True)[0].header,
                             naxis=2, relax=True).to_header(relax=True))
        self.set_header(hdr)

    def pix2world(self, xy, deg=True):
        """Convert pixel coordinates to celestial coordinates

        :param xy: A tuple (x,y) of pixel coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...)

        :param deg: (Optional) If true, celestial coordinates are
          returned in sexagesimal format (default False).

        .. note:: it is much more effficient to pass a list of
          coordinates than run the function for each couple of
          coordinates you want to transform.
        """
        xy = np.squeeze(xy).astype(float)
        if np.size(xy) == 2:
            x = [xy[0]]
            y = [xy[1]]
        elif np.size(xy) > 2 and len(xy.shape) == 2:
            if xy.shape[0] > xy.shape[1]:
                xy = np.copy(xy.T)
            x = xy[:,0]
            y = xy[:,1]
        else:
            raise StandardError('xy must be a tuple (x,y) of coordinates or a list of tuples ((x0,y0), (x1,y1), ...)')

        if not hasattr(self, 'dxmap') or not hasattr(self, 'dymap'):
            coords = np.array(
                self.wcs.all_pix2world(
                    x, y, 0)).T
        else:
            if np.size(x) == 1:
                xyarr = np.atleast_2d([x, y]).T
            else:
                xyarr = xy
                print xy
            coords = orb.utils.astrometry.pix2world(
                self.get_wcs_header(), self.dimx, self.dimy, xyarr, self.dxmap, self.dymap)
        if deg:
            return coords
        else: return np.array(
            [orb.utils.astrometry.deg2ra(coords[:,0]),
             orb.utils.astrometry.deg2dec(coords[:,1])])


    def world2pix(self, radec, deg=True):
        """Convert celestial coordinates to pixel coordinates

        :param xy: A tuple (x,y) of celestial coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...). Must be in degrees.

        .. note:: it is much more effficient to pass a list of
          coordinates than run the function for each couple of
          coordinates you want to transform.
        """
        radec = np.squeeze(radec)
        if np.size(radec) == 2:
            ra = [radec[0]]
            dec = [radec[1]]
        elif np.size(radec) > 2 and len(radec.shape) == 2:
            if radec.shape[0] < radec.shape[1]:
                radec = np.copy(radec.T)
            ra = radec[:,0]
            dec = radec[:,1]
        else:
            raise StandardError('radec must be a tuple (ra,dec) of coordinates or a list of tuples ((ra0,dec0), (ra1,dec1), ...)')

        if not hasattr(self, 'dxmap') or not hasattr(self, 'dymap'):
            coords = np.array(
                self.wcs.all_world2pix(
                    ra, dec, 0,
                    detect_divergence=False,
                    quiet=True)).T
        else:
            radecarr = np.atleast_2d([ra, dec]).T
            coords = orb.utils.astrometry.world2pix(
                self.get_wcs_header(), self.dimx, self.dimy, radecarr, self.dxmap, self.dymap)

        return coords

    def get_deep_frame(self):
        """Return deep frame if if exists. None if no deep frame is
        attached to the cube."""
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'deep_frame' in f:
                return f['deep_frame'][:]
            else: return None

    def get_header(self):
        """Return cube header."""
        if not hasattr(self, 'header'):
            header = self.get_cube_header()
            header['CTYPE3'] = 'WAVE-SIP' # avoid a warning for
                                          # inconsistency
            return copy.copy(header)
        else: return copy.copy(self.header)

    def set_header(self, hdr):
        """Set cube header"""
        if hdr is None: hdr = self.get_header()
        hdr['CTYPE3'] = 'WAVE-SIP' # avoid a warning for
                                   # inconsistency

        self.header = copy.copy(hdr)
        self.reset_params()

    def reset_params(self):
        """Read header again and reset parameters"""
        if not hasattr(self, 'header'): raise AttributeError('header attribute not set')

        self.set_param('step', float(self.header['STEP']))
        self.set_param('order', int(self.header['ORDER']))
        self.set_param('axis_corr', float(self.header['AXISCORR']))
        self.set_param('nm_laser', float(self.header['CALIBNM']))
        self.set_param('object_name', str(self.header['OBJECT']))
        self.set_param('filter_name', str(self.header['FILTER']))
        self.set_param('filter_file_path', self._get_filter_file_path(self.params.filter_name))
        self.set_param('apodization', float(self.header['APODIZ']))
        self.set_param('exposure_time', float(self.header['EXPTIME']))
        if 'FLAMBDA' in self.header:
            self.set_param('flambda', float(self.header['FLAMBDA']))
        else:
            warnings.warn('FLAMBDA keyword not in cube header. Flux calibration may be bad.')
            self.set_param('flambda', 1.)

        step_nb = int(self.header['STEPNB'])
        if step_nb != self.dimz:
            warnings.warn('Malformed spectral cube. The number of steps in the header ({}) does not correspond to the real size of the data cube ({})'.format(step_nb, self.dimz))
            step_nb = int(self.dimz)
        self.set_param('step_nb', step_nb)

        if 'ZPDINDEX' in self.header:
            self.set_param('zpd_index', self.header['ZPDINDEX'])
        else:
            raise KeyError('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')

        # new data prefix
        base_prefix = '{}_{}.{}'.format(self.params.object_name,
                                         self.params.filter_name,
                                         self.params.apodization)

        self._data_prefix = base_prefix + '.ORCS' + os.sep + base_prefix + '.'
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()

        # resolution
        resolution = orb.utils.spectrum.compute_resolution(
            self.dimz - self.params.zpd_index,
            self.params.step, self.params.order,
            self.params.axis_corr)
        self.set_param('resolution', resolution)

        # incident angle of reference (in degrees)
        self.set_param('theta_proj', orb.utils.spectrum.corr2theta(self.params.axis_corr))

        # wavenumber
        self.set_param('wavetype', str(self.header['WAVTYPE']))
        if self.params.wavetype == 'WAVELENGTH':
            raise Exception('ORCS cannot handle wavelength cubes')
            self.params['wavenumber'] = False
            logging.info('Cube is in WAVELENGTH (nm)')
            self.unit = 'nm'
        else:
            self.params['wavenumber'] = True
            logging.info('Cube is in WAVENUMBER (cm-1)')
            self.unit = 'cm-1'

        # wavelength calibration
        self.set_param('wavelength_calibration', bool(self.header['WAVCALIB']))

        if self.params.wavelength_calibration:
            logging.info('Cube is CALIBRATED in wavenumber')
        else:
            logging.info('Cube is NOT CALIBRATED')


        ## Get WCS header
        self.wcs = self.get_wcs()
        self.wcs_header = self.get_wcs_header()
        self._wcs_header = self.get_wcs_header()

        self.set_param('target_ra', float(self.wcs.wcs.crval[0]))
        self.set_param('target_dec', float(self.wcs.wcs.crval[1]))
        self.set_param('target_x', float(self.wcs.wcs.crpix[0]))
        self.set_param('target_y', float(self.wcs.wcs.crpix[1]))

        wcs_params = orb.utils.astrometry.get_wcs_parameters(self.wcs)
        self.set_param('wcs_rotation', float(wcs_params[-1]))

        self.set_param('obs_date', np.array(self.header['DATE-OBS'].strip().split('-'), dtype=int))
        if 'HOUR_UT' in self.header:
            self.set_param('hour_ut', np.array(self.header['HOUR_UT'].strip().split(':'), dtype=float))
        else:
            self.params['hour_ut'] = (0., 0., 0.)

        # create base axis of the data
        if self.params.wavenumber:
            self.set_param('base_axis', orb.utils.spectrum.create_cm1_axis(
                self.dimz, self.params.step, self.params.order,
                corr=self.params.axis_corr))
        else:
            self.set_param('base_axis', orb.utils.spectrum.create_nm_axis(
                self.dimz, self.params.step, self.params.order,
                corr=self.params.axis_corr))

        self.set_param('axis_min', np.min(self.params.base_axis))
        self.set_param('axis_max', np.max(self.params.base_axis))
        self.set_param('axis_step', np.min(self.params.base_axis[1] - self.params.base_axis[0]))
        self.set_param('line_fwhm', orb.utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index, self.params.step, self.params.order,
            apod_coeff=self.params.apodization,
            corr=self.params.axis_corr,
            wavenumber=self.params.wavenumber))
        self.set_param('filter_range', self.get_filter_range())


    def get_wcs(self):
        """Return the WCS of the cube as a astropy.wcs.WCS instance """
        return copy.copy(pywcs.WCS(self.get_header(), naxis=2, relax=True))

    def get_wcs_header(self):
        """Return the WCS of the cube as a astropy.wcs.WCS instance """
        return copy.copy(self.get_wcs().to_header(relax=True))


    def reproject(self):
        """Reproject data cube in a distorsion-less WCS.

        .. warning:: The amount of available RAM must be larger than
          the cube size on disk.
        """
        wcs = self.get_wcs()
        # removes automatically sip distortion
        new_wcs = pywcs.WCS(self.get_wcs().to_header())
        X, Y = np.mgrid[:2048,:2064]
        XYp = wcs.all_world2pix(
            new_wcs.all_pix2world(
                np.array([X.flatten(),Y.flatten()]).T,0), 0)
        Xp, Yp = XYp.T

        reprojected_cube = np.empty((self.dimx, self.dimy, self.dimz),
                                    dtype=np.float32)
        progress = orb.core.ProgressBar(self.dimz)
        for i in range(self.dimz):
            progress.update(i)
            idat = self.get_data_frame(i, silent=True)
            idatf = scipy.interpolate.RectBivariateSpline(
                np.arange(idat.shape[0]),
                np.arange(idat.shape[1]),
                idat, kx=1, ky=1, s=0)
            reprojected_cube[:,:,i] = idatf.ev(
                Xp.reshape(*idat.shape),
                Yp.reshape(*idat.shape))
        progress.end()
        self.write_fits(self._get_reprojected_cube_path(), reprojected_cube,
                        overwrite=True)

    def get_flux_uncertainty(self):
        """Return the uncertainty on the flux at a given wavenumber in
        erg/cm2/s/channel. It corresponds to the uncertainty (1 sigma)
        of the spectrum in a given channel.

        :param wavenumber: Wavenumber (cm-1)
        """
        deep_frame = self.get_deep_frame()
        if deep_frame is None:
            warnings.warn("No deep frame in the HDF5 cube. Please use a cube reduced with the last version of ORBS")
            return None

        # compute counts/s
        # total number of counts in a full cube
        total_counts = deep_frame * self.dimz

        # associated photon noise distributed over each channel in counts
        noise_counts = np.sqrt(total_counts)

        # in flux unit
        noise_flux = noise_counts / self.params.exposure_time # counts/s
        noise_flux *= self.params.flambda / self.dimz # erg/cm2/s/A
        # compute mean channel size
        channel_size_ang = 10 * orb.utils.spectrum.fwhm_cm12nm(
            np.diff(self.params.base_axis)[0],
            self.params.base_axis[0] + np.diff(self.params.base_axis)[0]/ 2)

        noise_flux *= channel_size_ang * orb.constants.FWHM_SINC_COEFF # erg/cm2/s/channel

        return noise_flux


    def ivisual(self):
        """Plot an interactive view of the deep frame wich offers the
        possibility to select and draw spectra.

        This can be used inside a jupyter session. Works best with
        `%matplotlib notebook` magic function.
        """
        import pylab as pl
        from matplotlib.widgets import Slider, Button
        from matplotlib import gridspec

        def show_df(vmin=None, vmax=None):
            df = self.get_deep_frame().T
            if vmin is None and vmax is None and self.imshown is not None:
                vmin, vmax = self.imshown.get_clim()
            self.imshown = self.axes[0].imshow(
                df, origin='bottom-left', vmin=vmin, vmax=vmax)

        def onclick(event):
            if event.button == 3 and event.inaxes == self.axes[0]:
                # get data
                axis, spec = self.extract_spectrum(
                    event.xdata, event.ydata, self.radius)
                self.axes[1].cla()
                self.axes[1].plot(axis, spec, label='{:.1f} {:.1f}'.format(
                    event.xdata, event.ydata))
                self.axes[1].grid()
                self.axes[1].set_xlabel(r'Wavenumber (cm$^{-1}$)')
                self.axes[1].set_ylabel(r'Flux (erg/cm$^{2}$/s/$\AA$)')
                self.axes[1].legend()
                self.xlim = self.axes[0].get_xlim()
                self.ylim = self.axes[0].get_ylim()
                self.axes[0].cla()
                show_df()
                self.axes[0].set_xlim(self.xlim)
                self.axes[0].set_ylim(self.ylim)
                self.axes[0].axvline(x=event.xdata, c='orange', alpha=0.7)
                self.axes[0].axhline(y=event.ydata, c='orange', alpha=0.7)
                circle = pl.Circle((event.xdata, event.ydata), self.radius,
                                   color='orange', fill=False, alpha=0.7)
                self.axes[0].add_artist(circle)

        def radius_update(val):
            self.radius = float(val)

        def norm(_):
            self.xlim = self.axes[0].get_xlim()
            self.ylim = self.axes[0].get_ylim()
            df = self.get_deep_frame().T
            box = df[int(self.xlim[0]):int(self.xlim[1]),
                     int(self.ylim[0]):int(self.ylim[1])]
            vmin = np.nanpercentile(box, 1)
            vmax = np.nanpercentile(box, 99)
            show_df(vmin=vmin, vmax=vmax)

        self.fig = pl.figure(figsize=(10, 7))
        self.imshown = None
        gs  = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 0.05])
        ax0 = pl.subplot(gs[0])
        ax1 = pl.subplot(gs[1])
        ax2 = pl.subplot(gs[2])
        ax3 = pl.subplot(gs[4])

        self.axes = (ax0, ax1, ax2, ax3)
        self.xlim = None
        self.ylim = None
        show_df()
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.radius = 10
        self.sradius = Slider(self.axes[2], 'Radius', 1, 30.0, valinit=self.radius)
        self.sradius.on_changed(radius_update)
        self.bnorm = Button(self.axes[3], 'Normalize Colorbar')
        self.bnorm.on_clicked(norm)

    def get_heliocentric_velocity(self, vobs):
        """Return corrected heliocentric velocity and local standard of rest
           velocity for a given measured velocity.
        """
        rvcorr = rvcorrect.RVCorrect(
            self.params.target_ra, self.params.target_dec, self.params.obs_date,
            self.params.hour_ut, (self.config.OBS_LAT, self.config.OBS_LON, self.config.OBS_ALT))
        return rvcorr.rvcorrect(vobs=vobs)


##################################################
#### CLASS CubeJobServer #########################
##################################################

class CubeJobServer(object):

    def __init__(self, cube):
        self.job_server, self.ncpus = orb.utils.parallel.init_pp_server()
        self.cube = cube


    def process_by_region(self, func, regions, subtract, axis, args=list(), modules=list(),
                          depfuncs=list()):

        """Parallelize a function applied to a list of intergrated
        regions extracted from the spectral cube.

        the function must be func(spectrum, theta_orig, *args)

        theta_orig is the mean original incident angle in the integrated region.
        """
        self.all_jobs = [(i, regions[i]) for i in range(len(regions))] # jobs to submit

        # jobs submit / retrieve loop
        out = list()
        self.jobs = list() # submitted and unfinished jobs
        all_jobs_nb = len(self.all_jobs)
        progress = orb.core.ProgressBar(all_jobs_nb)
        while len(self.all_jobs) > 0 or len(self.jobs) > 0:
            while_loop_start = time.time()

            # submit jobs
            while len(self.jobs) < self.ncpus and len(self.all_jobs) > 0:
                timer = dict()
                timer['job_submit_start'] = time.time()

                timer['job_load_data_start'] = time.time()
                # raw lines extraction (warning: velocity must be
                # corrected by the function itself)
                ispectrum, itheta_orig = self.cube._extract_spectrum_from_region(
                    self.all_jobs[0][1],
                    subtract_spectrum=subtract,
                    silent=True, output_axis=axis, return_mean_theta=True)

                timer['job_load_data_end'] = time.time()

                all_args = list()
                all_args.append(ispectrum)
                all_args.append(itheta_orig)
                for iarg in args:
                    all_args.append(iarg)

                timer['job_submit_end'] = time.time()

                # job submission
                self.jobs.append([
                    self.job_server.submit(
                        func,
                        args=tuple(all_args),
                        modules=tuple(modules),
                        depfuncs=tuple(depfuncs)),
                    self.all_jobs[0], time.time(), timer])
                self.all_jobs.pop(0)
                progress.update(all_jobs_nb - len(self.all_jobs))


            # retrieve all finished jobs
            unfinished_jobs = list()
            for i in range(len(self.jobs)):
                ijob, (iregion_index, iregion), stime, timer = self.jobs[i]
                if ijob.finished:

                    logging.debug('job time since submission: {} s'.format(
                        time.time() - stime))
                    logging.debug('job submit time: {} s'.format(
                        timer['job_submit_end'] - timer['job_submit_start']))
                    logging.debug('job load data time: {} s'.format(
                        timer['job_load_data_end'] - timer['job_load_data_start']))

                    out.append((iregion_index, ijob(), ispectrum))
                    logging.debug('job time (whole loop): {} s'.format(time.time() - stime))
                else:
                    unfinished_jobs.append(self.jobs[i])
            self.jobs = unfinished_jobs


        progress.end()

        orb.utils.parallel.close_pp_server(self.job_server)

        # reorder out
        ordered_out = list()
        for i in range(all_jobs_nb):
            ok = False
            for iout in out:
                if iout[0] == i:
                    ordered_out.append(iout[1:])
                    ok = True
                    break
            if not ok:
                raise StandardError('at least one of the processed region is not in the results list')

        return ordered_out


    def process_by_pixel(self, func, args=list(), modules=list(), out=dict(),
                         depfuncs=list(),
                         mask=None, binning=1):
        """Parallelize a function taking binned spectra of the cube as
        an input. All pixels are gone through unless a mask is passed
        which indicates the pixels that must be processed. The typical
        results returned are maps.

        The function must be func(spectrum, *args)

        Any argument with a shape equal to the x,y shape of the cube
        (or the binned x,y shape) will be mapped, i.e., the argument
        passed to the vector function will be the value corresponding
        to the position of the extracted spectrum. (works also for 3d
        shaped arguments, the 3rd dimension can have any size)
        """

        def process_in_line(*args):
            """Basic line processing for a vector function"""
            import marshal, types
            import numpy as np
            ## function is unpicked
            _code = marshal.loads(args[0])
            _func = types.FunctionType(_code, globals(), '_func')
            iline_data = np.squeeze(args[1])
            iline_data = np.atleast_2d(iline_data)
            out_line = list()
            for i in range(iline_data.shape[0]):
                iargs_list = list()
                # remap arguments
                for iarg in args[2:]:
                    try:
                        shape = iarg.shape
                    except AttributeError:
                        shape = None
                    if shape is not None:
                        iarg = np.squeeze(iarg)
                        shape = iarg.shape
                        if shape == (iline_data.shape[0],):
                            iarg = iarg[i]

                    iargs_list.append(iarg)
                try:
                    out_line.append(_func(iline_data[i,:], *iargs_list))
                except Exception, e:
                    out_line.append(None)
                    logging.warning('Exception occured in process_in_line at function call level: {}'.format(e))

            return out_line


        ## function must be serialized (or picked)
        func = marshal.dumps(func.func_code)

        binning = int(binning)

        binned_shape = orb.utils.image.nanbin_image(
            np.ones((self.cube.dimx, self.cube.dimy)),
            int(binning)).shape

        def isbinned(_data):
            if (_data.shape[0] == self.cube.dimx
                and _data.shape[1] == self.cube.dimy):
                return False
            elif (_data.shape[0] == binned_shape[0]
                  and _data.shape[1] == binned_shape[1]):
                return True
            else: raise Exception('Strange data shape {}. Must be correctly binned ({}, {}) or unbinned ({}, {})'.format(_data.shape, binned_shape[0], binned_shape[1], self.cube.dimx, self.cube.dimy))


        # check outfile
        self.out_is_dict = True
        if not isinstance(out, dict):
            self.out_is_dict = False
            if not isinstance(out, np.ndarray):
                raise TypeError('out must be a numpy.ndarray')
            elif out.ndim < 2:
                raise ValueError('out must be at least a 2d numpy.ndarray')
            elif out.shape[0] != int(self.cube.dimx)/int(binning) or out.shape[1] != int(self.cube.dimy)/int(binning):
                raise ValueError(
                    'out.shape[0] and out.shape[1] must be equal to cube.dimx and cube.dimy')

        # check mask
        if not mask is None:
            try:
                mask.shape == (self.cube.dimx, self.cube.dimy)
            except TypeError:
                raise TypeError('mask must be a numpy.ndarray')
            except Exception, e:
                logging.error('Bad mask format{}', e)

        else:
            mask = np.ones((self.cube.dimx, self.cube.dimy), dtype=bool)

        if binning > 1:
            if not isbinned(mask):
                mask = orb.utils.image.nanbin_image(mask, int(binning))

        # check arguments
        # reshape passed arguments
        for i in range(len(args)):
            new_arg = args[i]
            is_map = False
            try:
                shape = new_arg.shape
            except AttributeError:
                shape = None
            except KeyError:
                shape = None

            if shape is not None:
                if not isbinned(new_arg) and new_arg.ndim == 2:
                    new_arg = orb.utils.image.nanbin_image(new_arg, int(binning))
                    is_map = True
                elif isbinned(new_arg):
                    is_map = True
                else:
                    raise TypeError('Data shape {} not handled'.format(new_arg.shape))

            args[i] = (new_arg, is_map)

        # get pixel positions grouped by line
        xy = list()
        for i in range(mask.shape[1]):
            _X = np.nonzero(mask[:,i])[0]
            if len(_X) > 0:
                xy.append((_X, np.ones(len(_X), dtype=np.int64) * i))
        logging.info('{} lines to fit'.format(len(xy)))

        # jobs will be passed by line
        self.all_jobs_indexes = range(len(xy))
        all_jobs_nb = len(self.all_jobs_indexes)

        # jobs submit / retrieve loop
        self.jobs = list()
        progress = orb.core.ProgressBar(all_jobs_nb)
        while len(self.all_jobs_indexes) > 0 or len(self.jobs) > 0:
            while_loop_start = time.time()

            # submit jobs
            while len(self.jobs) < self.ncpus and len(self.all_jobs_indexes) > 0:
                timer = dict()
                timer['job_submit_start'] = time.time()

                ix, iy = xy[self.all_jobs_indexes[0]]

                timer['job_load_data_start'] = time.time()

                # raw lines extraction (warning: velocity must be
                # corrected by the function itself)

                iline = self.cube.get_data(min(ix) * binning, (max(ix) + 1) * binning,
                                           iy[0] * binning, (iy[0] + 1) * binning,
                                           0, self.cube.dimz, silent=True)

                if binning > 1:
                    iline = orb.utils.image.nanbin_image(iline, binning) * binning**2

                iline = np.atleast_2d(iline)
                iline = iline[ix - min(ix), :]

                timer['job_load_data_end'] = time.time()

                all_args = list()
                all_args.append(func)
                all_args.append(iline)

                # extract values of mapped arguments
                for iarg in args:
                    if iarg[1]:
                        all_args.append(np.copy(iarg[0][ix, iy, ...]))
                    else:
                        all_args.append(iarg[0])

                timer['job_submit_end'] = time.time()

                # job submission
                self.jobs.append([
                    self.job_server.submit(
                        process_in_line,
                        args=tuple(all_args),
                        modules=tuple(modules),
                        depfuncs=tuple(depfuncs)),
                    (ix, iy), time.time(), timer])
                self.all_jobs_indexes.pop(0)
                progress.update(all_jobs_nb - len(self.all_jobs_indexes))


            # retrieve all finished jobs
            unfinished_jobs = list()
            for i in range(len(self.jobs)):
                ijob, (ix, iy), stime, timer = self.jobs[i]
                if ijob.finished:

                    logging.debug('job time since submission: {} s'.format(
                        time.time() - stime))
                    logging.debug('job submit time: {} s'.format(
                        timer['job_submit_end'] - timer['job_submit_start']))
                    logging.debug('job load data time: {} s'.format(
                        timer['job_load_data_end'] - timer['job_load_data_start']))

                    res_line = ijob()
                    for iline in range(len(res_line)):
                        res = res_line[iline]
                        if self.out_is_dict:
                            if not isinstance(res, dict):
                                raise TypeError('function result must be a dict if out is a dict')
                            for ikey in res.keys():
                                # create the output array if not set
                                if ikey not in out and res[ikey] is not None:
                                    if np.size(res[ikey]) > 1:
                                        if res[ikey].ndim > 1: raise TypeError('must be a 1d array of floats')
                                        try: float(res[ikey][0])
                                        except TypeError: raise TypeError('must be an array of floats')
                                    else:
                                        try:
                                            float(res[ikey])
                                        except TypeError:
                                            raise TypeError('If out dict maps are not set (i.e. out is set to a default dict()) returned values must be a dict of float or a 1d array of floats')
                                    _iout = np.empty(
                                        (self.cube.dimx/binning,
                                         self.cube.dimy/binning,
                                         np.size(res[ikey])),
                                        dtype=float)

                                    _iout = np.squeeze(_iout)
                                    out[ikey] = _iout
                                    out[ikey].fill(np.nan)

                                if res[ikey] is not None:
                                    out[ikey][ix[iline], iy[iline], ...] = res[ikey]
                        else:
                            out[ix[iline], iy[iline], ...] = ijob()
                    logging.debug('job time (whole loop): {} s'.format(time.time() - stime))
                else:
                    unfinished_jobs.append(self.jobs[i])
            self.jobs = unfinished_jobs


            #logging.debug('while loop time: {} s'.format(time.time() - while_loop_start))
        progress.end()

        orb.utils.parallel.close_pp_server(self.job_server)

        return out

    def __del__(self):
        try:
            orb.utils.parallel.close_pp_server(self.job_server)
        except IOError: pass




##################################################
#### CLASS LineMaps ##############################
##################################################

class LineMaps(orb.core.Tools):
    """Manage line parameters maps"""


    lineparams = ('height', 'height-err', 'amplitude', 'amplitude-err',
                  'velocity', 'velocity-err', 'fwhm', 'fwhm-err',
                  'sigma', 'sigma-err', 'flux', 'flux-err',
                  'logGBF', 'chi2', 'rchi2', 'ks_pvalue')


    def __init__(self, dimx, dimy, lines, wavenumber, binning, div_nb,
                 project_header=None, wcs_header=None, **kwargs):
        """Init class

        :param dimx: X dimension of the unbinned data

        :param dimy: Y dimension of the unbinned data

        :param lines: tuple of the line names

        :param wavenumber: True if the data is in wavenumber, False if
          it is in wavelength.

        :param binning: Binning of the data.

        :param div_nb: Number of divisions if the data is binned in quadrant mode.

        :param project_header: (Optional) FITS header passed to the
          written frames (default None).

        :param wcs_header: (Optional) WCS header passed to the written
          frames (default None).

        :param kwargs: Kwargs are :meth:`~core.Tools.__init__` kwargs.
        """
        orb.core.Tools.__init__(self, **kwargs)
        self._project_header = project_header
        self.wcs_header = wcs_header
        self.__version__ = version.__version__

        self.wavenumber = wavenumber
        self.div_nb = div_nb
        self.binning = binning

        if binning > 1:
            # not optimal but always returns the exact numbers
            self.dimx, self.dimy = orb.utils.image.nanbin_image(
                np.empty((dimx, dimy), dtype=float), binning).shape
        else:
            self.dimx = int(dimx)
            self.dimy = int(dimy)

        self.unbinned_dimx = int(dimx)
        self.unbinned_dimy = int(dimy)


        # Create dataset
        if np.size(lines) == 1:
            self.lines = np.array([np.squeeze(lines)])
        else:
            self.lines = lines

        if self.wavenumber:
            self.line_names = orb.core.Lines().round_nm2ang(
                orb.utils.spectrum.cm12nm(self.lines))
            self.unit = 'cm-1'
        else:
            self.line_names = orb.core.Lines().round_nm2ang(
                self.lines)
            self.unit = 'nm'

        if np.size(self.line_names) == 1:
            self.line_names = np.array([np.squeeze(self.line_names)])

        # manage lines with same name
        _line_names = list()
        for line in self.line_names:
            test_line = str(line)
            index = 2
            while test_line in _line_names:
                test_line = str(line) + '_{}'.format(index)
                index += 1
            _line_names.append(test_line)
        self.line_names = _line_names

        self.data = dict()
        base_array =  np.empty((self.dimx, self.dimy, len(lines)),
                               dtype=float)
        base_array.fill(np.nan)
        for iparam in self.lineparams:
            self.data[iparam] = np.copy(base_array)

        # load computed maps
        self._load_maps()


    def _get_map_path(self, line_name, param, binning=None):
        """Return the path to a map of one gaussian fit parameter for
        one given emission line.

        :param line_name: Name of the emission line

        :param param: Parameter name

        :param binning: (Optional) Binning of the map. If not given
          instance binning is used (default None).
        """
        if binning is None:
            binning = self.binning

        if param not in self.lineparams:
            raise StandardError('Bad parameter')

        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        if line_name is not None:
            line_str = '.{}'.format(line_name)
        else:
            line_str = '.all'
        return (dirname + os.sep + "MAPS" + os.sep
                + basename + "map{}.{}x{}.{}.fits".format(
                    line_str, binning, binning, param))


    def _get_map_header(self, file_type, comment=None):
        """Return map header

        :param file_type: Type of file

        :param comment: (Optional) Comments on the file type (default
          None).
        """
        hdr = (self._get_basic_header(file_type)
               + self._project_header
               + self._get_basic_frame_header(self.dimx, self.dimy))
        hdr = self._add_wcs_header(hdr)
        hdr = orb.core.Header(hdr)
        hdr.bin_wcs(self.binning)
        return hdr


    def _add_wcs_header(self, hdr):
        """Add WCS header keywords to a header.

        :param hdr: Header to update
        """
        if self.wcs_header is not None:
            new_hdr = pyfits.Header()
            new_hdr.extend(hdr, strip=True,
                           update=True, end=True)

            new_hdr.extend(self.wcs_header, strip=True,
                           update=True, end=True)

            if 'RESTFRQ' in new_hdr: del new_hdr['RESTFRQ']
            if 'RESTWAV' in new_hdr: del new_hdr['RESTWAV']
            if 'LONPOLE' in new_hdr: del new_hdr['LONPOLE']
            if 'LATPOLE' in new_hdr: del new_hdr['LATPOLE']

            return new_hdr
        else:
            return hdr

    def _load_maps(self):
        """Load already computed maps with the smallest binning but
        still higher than requested. Loaded maps can be used to get
        initial fitting parameters."""
        # check existing files
        binnings = np.arange(self.binning+1, 1000)
        available_binnings = list()
        for binning in binnings:
            all_ok = True
            for line_name in self.line_names:
                for param in self.lineparams:
                    if not os.path.exists(self._get_map_path(
                        line_name, param, binning)):
                        all_ok = False
            if all_ok: available_binnings.append(binning)
        if len(available_binnings) < 1: return
        # load data from lowest (but still higher than requested)
        # binning
        binning = np.nanmin(available_binnings)
        logging.info('Loading {}x{} maps'.format(
            binning, binning))
        for param in self.lineparams:
            # only velocity param is loaded
            if param in ['velocity', 'velocity-err', 'sigma', 'sigma-err']:
                data = np.empty(
                    (self.dimx, self.dimy, len(self.lines)),
                    dtype=float)
                data.fill(np.nan)
                for iline in range(len(self.lines)):
                    map_path = self._get_map_path(
                        self.line_names[iline], param, binning)
                    old_map = self.read_fits(map_path)
                    real_old_map = np.copy(old_map)
                    # data is unbinned and rebinned : creates small
                    # errors, but loaded maps are only used for initial
                    # parameters
                    old_map = orb.cutils.unbin_image(
                        old_map,
                        self.unbinned_dimx,
                        self.unbinned_dimy)
                    old_map = orb.cutils.nanbin_image(
                        old_map, self.binning)
                    old_map = old_map[:self.dimx,:self.dimy]

                    data[:,:,iline] = np.copy(old_map)
                logging.info('{} loaded'.format(param))
                self.set_map(param, data)



    def set_map(self, param, data_map, x_range=None, y_range=None):
        """Set map values.

        :param param: Parameter

        :param data_map: Data

        :param x_range: (Optional) Data range along X axis (default
          None)

        :param y_range: (Optional) Data range along Y axis (default
          None)
        """
        if not isinstance(data_map, np.ndarray):
            raise TypeError('data_map  must be a numpy.ndarray')

        if (data_map.shape[0] != self.data[param].shape[0]
            or data_map.shape[1] != self.data[param].shape[1]):
            raise TypeError('data_map must has the wrong size')
        if data_map.ndim > 3:
            raise TypeError('data_map must have 2 or 3 dimensions')

        if param not in self.lineparams:
            raise StandardError('Bad parameter')

        if x_range is None and y_range is None:
            self.data[param] = data_map
        else:
            if data_map.ndim == 3:
                self.data[param][
                    min(x_range):max(x_range),
                    min(y_range):max(y_range), :] = data_map
            else:
                for ik in range(self.data[param].shape[2]):
                    self.data[param][
                        min(x_range):max(x_range),
                        min(y_range):max(y_range), ik] = data_map

    def get_map(self, param, x_range=None, y_range=None):
        """Get map values

        :param param: Parameter

        :param x_range: (Optional) Data range along X axis (default
          None)

        :param y_range: (Optional) Data range along Y axis (default
          None)
        """

        if x_range is None:
            x_range = [0, self.dimx]
        if y_range is None:
            y_range = [0, self.dimy]

        return self.data[param][
            x_range[0]:x_range[1],
            y_range[0]:y_range[1]]

    def write_maps(self):
        """Write all maps to disk."""

        for param in self.lineparams:

            logging.info('Writing {} maps'.format(param))

            if 'fwhm' in param:
                unit = ' [in {}]'.format(self.unit)
            elif 'velocity' in param:
                unit = ' [in km/s]'
            else: unit = ''


            # check if data is the same for all the lines
            same_param = True
            if len(self.lines) > 1:
                if np.all(np.isnan(self.data[param])):
                    same_param = False
                else:
                    for icheck in range(1, len(self.lines)):
                        nonans = np.nonzero(~np.isnan(self.data[param][:,:,0]))
                        if np.any(self.data[param][:,:,0][nonans] != self.data[param][:,:,icheck][nonans]):
                            same_param = False
                            break

            if same_param:
                logging.warning('param {} is the same for all lines'.format(param))
                lines = list(['fake'])
            else:
                lines = list(self.lines)

            for iline in range(len(lines)):
                if not same_param:
                    line_name = self.line_names[iline]
                else:
                    line_name = None

                new_map = self.data[param][:,:,iline]

                map_path = self._get_map_path(
                    line_name, param=param)

                # load old map if it exists
                if os.path.exists(map_path):
                    old_map = self.read_fits(map_path)
                    nonans = np.nonzero(~np.isnan(new_map))
                    old_map[nonans] = new_map[nonans]
                    new_map = old_map

                self.write_fits(
                    map_path, new_map,
                    overwrite=True,
                    fits_header=self._get_map_header(
                        "Map {} {}{}".format(
                            param, line_name, unit)))

                if same_param: break



class Filter(object):

    def __init__(self):
        pass
