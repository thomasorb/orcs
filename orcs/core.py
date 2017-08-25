#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
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
    def __init__(self, cube_path, **kwargs):
        """
        :param cube_path: Path to the HDF5 cube.

        :param kwargs: Kwargs are :meth:`orb.core.HDFCube` properties.
        """
        FIT_TOL = 1e-10
        self.cube_path = cube_path
        self.header = self.get_cube_header()
        instrument = None
        if 'SITELLE' in self.header['INSTRUME']:
            instrument = 'sitelle'

        kwargs['instrument'] = instrument
        orb.core.HDFCube.__init__(self, cube_path, **kwargs)

        self.overwrite = True
        
        self.set_param('init_fwhm', float(self._get_config_parameter('INIT_FWHM')))
        self.set_param('fov', float(self._get_config_parameter('FIELD_OF_VIEW_1')))
        self.set_param('init_wcs_rotation', float(self._get_config_parameter('INIT_ANGLE')))
        

        self.fit_tol = FIT_TOL
        
        self.set_param('step', float(self.header['STEP']))
        self.set_param('order', int(self.header['ORDER']))
        self.set_param('axis_corr', float(self.header['AXISCORR']))
        self.set_param('nm_laser', float(self.header['CALIBNM']))
        self.set_param('object_name', str(self.header['OBJECT']))
        self.set_param('filter_name', str(self.header['FILTER']))
        self.set_param('filter_file_path', self._get_filter_file_path(self.params.filter_name))
        self.set_param('apodization', float(self.header['APODIZ']))
        self.set_param('step_nb', int(self.header['STEPNB']))
        if self.params.step_nb != self.dimz:
            self._print_error('Malformed spectral cube. The number of steps in the header ({}) does not correspond to the real size of the data cube ({})'.format(self.params.step_nb, self.dimz))
        if 'ZPDINDEX' in self.header:
            self.set_param('zpd_index', self.header['ZPDINDEX'])
        else:
            self._print_error('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')


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
        self.set_param('theta', np.rad2deg(np.arccos(1./self.params.axis_corr)))

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
            logging.info('Cube is CALIBRATED')
        else:
            logging.info('Cube is NOT CALIBRATED')            
            

        ## Get WCS header
        self.header['CTYPE3'] = 'WAVE-SIP' # avoid a warning for
                                           # inconsistency
        self.wcs = pywcs.WCS(self.header, naxis=2, relax=True)
        self.wcs_header = self.wcs.to_header(relax=True)
        self._wcs_header = pywcs.WCS(self.wcs_header, naxis=2, relax=True)

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
        
    def _get_data_prefix(self):
        """Return data prefix"""
        return self._data_prefix

    def _parallel_process_in_quads(self, func, args, modules,
                                   out_cube_path,
                                   binning=1, vector=True):

        """
        General parallel quadrant-based processing.

        Built to process a cube in columns or vector by vector and
        write the processed cube as an HDF5 quad cube.

        The process is split over a certain number of quadrants to
        avoid loading a full data cube in memory.

        :param func: Can be a vector or column function (vector
          must be set to False in the case of a function applied to
          the columns). Must be `f(vector_data, *args)` or
          `f(column_data, *args)`.
    
        :param args: Function arguments. All arguments with a 2D shape
          equal to the cube 2D shape are treated as mapped
          arguments. The column or pixel corresponding to the column
          or pixel computed is extracted.

        :param modules: Modules to pass to each process.

        :param out_cube_path: Path to the output resulting cube.
        
        :param vector: (Optional) if vector is True, the function is
          applied to each vector (each pixel) of the cube. Else the
          function is applied to each column of the cube (default
          True)

        :param binning: (Optional) 'on-the-fly' binning of the data
          (warning: the ouput data shape is binned) (default 1)
        """
        def process_in_column(args):
            """Basic column processing for a vector function"""
            import marshal, types
            import numpy as np
            ## function is unpicked
            _code = marshal.loads(args[0])
            _func = types.FunctionType(_code, globals(), '_func')
            icolumn_data = np.squeeze(args[1])
            for i in range(icolumn_data.shape[0]):
                iargs_list = list()
                for iarg in args[2:]:
                    try:
                        shape = iarg.shape
                    except AttributeError:
                        shape = None
                    if shape is not None:
                        iarg = np.squeeze(iarg)
                        shape = iarg.shape
                        if shape == (icolumn_data.shape[0],):
                            iarg = iarg[i]
                            
                    iargs_list.append(iarg)
                        
                icolumn_data[i,:] = _func(icolumn_data[i,:], *iargs_list)
            
            return icolumn_data

        ## function must be serialized (or picked)
        func = marshal.dumps(func.func_code)
        
        if vector:
            col_func = process_in_column
        else:
            col_func = func

        out_cube = orb.core.OutHDFQuadCube(
            out_cube_path,
            (self.dimx / binning, self.dimy / binning, self.dimz),
            self.config.QUAD_NB,
            reset=True)


        for iquad in range(0, self.config.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                       0, self.dimz)
            ## iquad_data = np.zeros((x_max - x_min, y_max - y_min, self.dimz), dtype=float)
        

            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = orb.core.ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus * binning):
                progress.update(ii, 'processing quad{}/{}'.format(
                    iquad+1, self.config.QUAD_NB))
                # no more jobs than columns
                if (ii + ncpus * binning >= x_max - x_min): 
                    ncpus = (x_max - x_min - ii) / binning

                # construct passed arguments
                jobs_args_list = list()
                for ijob in range(ncpus):
                    iargs_list = list()
                    
                    if vector:
                        iargs_list.append(func)                        
                    else:
                        iargs_list.append(None)

                    icolumn_data = iquad_data[
                        ii + ijob * binning:
                        ii + (ijob+1) * binning,:,:]
                    
                    iargs_list.append(icolumn_data)
                        
                    for iarg in args:
                        try:
                           shape = iarg.shape
                        except AttributeError:
                            shape = None
                        if shape is not None:
                            if shape == (self.dimx, self.dimy):
                                icolumn = iarg[
                                    x_min + ii + ijob * binning:
                                    x_min + ii + (ijob+1) * binning,
                                    y_min:y_max]
                    
                                # reduce column to binning
                                if binning > 1:
                                    icolumn = orb.utils.image.nanbin_image(
                                        icolumn, binning)
                            else:
                                icolumn = iarg
                            iargs_list.append(icolumn)
                        else:
                            iargs_list.append(iarg)
                    jobs_args_list.append(iargs_list)                
                    
                # jobs submission
                jobs = [(ijob, job_server.submit(
                    process_in_column,
                    args=(jobs_args_list[ijob],),
                    modules=modules))
                        for ijob in range(ncpus)]

                # get results
                for ijob, job in jobs:
                    icol_res = job()
                    iquad_data[ii+ijob,:icol_res.shape[0], :] = icol_res
                    
            progress.end()
            
            # save data
            logging.info('Writing quad {}/{} to disk'.format(
                iquad+1, self.config.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(iquad, data=iquad_data)
            logging.info('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.config.QUAD_NB, time.time() - write_start_time))

        out_cube.close()
        del out_cube


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
                                      return_spec_nb=False):
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
          
        :return: A scipy.UnivariateSpline object.
        """
        def _interpolate_spectrum(spec, corr, wavenumber, step, order, base_axis):
            if wavenumber:
                corr_axis = orb.utils.spectrum.create_cm1_axis(
                    spec.shape[0], step, order, corr=corr)
                return orb.utils.vector.interpolate_axis(
                    spec, base_axis, 5, old_axis=corr_axis)
            else:
                corr_axis = orb.utils.spectrum.create_nm_axis(
                    spec.shape[0], step, order, corr=corr)
                return orb.utils.vector.interpolate_axis(
                    spec, base_axis, 5, old_axis=corr_axis)
            
            
        def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
                                        median,
                                        wavenumber, base_axis, step, order):

            for icol in range(data_col.shape[0]):
                if mask_col[icol]:
                    corr = calib_coeff_col[icol]
                    data_col[icol, :] = _interpolate_spectrum(
                        data_col[icol, :], corr, wavenumber, step, order, base_axis)
                    ####
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

        if np.sum(mask) == 0: self._print_error('A region must contain at least one valid pixel')
        
        elif np.sum(mask) == 1:
            ii = region[0][0] ; ij = region[1][0]
            spectrum = _interpolate_spectrum(
                self[ii, ij, :], calibration_coeff_map[ii, ij],
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
                              self.params.order), 
                        modules=('import numpy as np',
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
                    
        if subtract_spectrum is not None:
            spectrum -= subtract_spectrum * counts

        if mean_flux:
            spectrum /= counts

        spectrum_function = scipy.interpolate.UnivariateSpline(
            self.params.base_axis[~np.isnan(spectrum)], spectrum[~np.isnan(spectrum)],
            s=0, k=1, ext=1)

        if return_spec_nb:
            return spectrum_function, counts
        else:
            return spectrum_function

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

        :param snr_guess: (Optional) Can only be None (classical fit)
          or 'auto' (Bayesian fit).

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
        if snr_guess not in ('auto', None):
            raise ValueError("snr_guess must be 'auto' or None")

        ## check if input params is instanciated
        if not hasattr(self, 'inputparams'):
            self._print_error('Input params not defined')

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
         
        fwhm_map = gvar.mean(self.inputparams.allparams['fwhm_guess']
                             * self.get_calibration_coeff_map_orig()
                             / self.get_calibration_coeff_map())
        fwhm_map = orb.utils.image.nanbin_image(fwhm_map, binning)
        
        mask = np.zeros((self.dimx, self.dimy), dtype=float)
        mask[region] = 1

        mask_bin = orb.utils.image.nanbin_image(mask, binning)

        for ii in range(mask_bin.shape[0]):
            for ij in range(mask_bin.shape[1]):
                unbinned_ii = ii*binning
                unbinned_ij = ij*binning
                if mask_bin[ii, ij]:
                    # load spectrum
                    axis, spectrum = self.extract_spectrum_bin(
                        unbinned_ii, unbinned_ij, binning,
                        subtract_spectrum=subtract_spectrum)

                    # check init velocity (already computed from
                    # binned maps). If a guess exists, velocity range
                    # is set to None
                    if not np.isnan(init_velocity_map[ii, ij]):
                        shift_guess_sdev = gvar.sdev(
                            inputparams.allparams['pos_cov'])
                        shift_guess = gvar.gvar(
                            np.ones_like(shift_guess_sdev) * init_velocity_map[ii, ij],
                            shift_guess_sdev)
                    else:
                        shift_guess = None

                    # check init sigma (already computed from binned
                    # maps).
                    if not np.isnan(init_sigma_map[ii, ij]):
                        sigma_guess_sdev = gvar.sdev(
                            inputparams.allparams['sigma_cov'])
                        sigma_guess = gvar.gvar(
                            np.ones_like(sigma_guess_sdev) * init_sigma_map[ii, ij],
                            sigma_guess_sdev)
                    else:
                        sigma_guess = None
                        
                    ifit = self._fit_lines_in_spectrum(
                        spectrum, snr_guess=snr_guess, sigma_guess=sigma_guess,
                        pos_cov=shift_guess, fwhm_guess=fwhm_map[ii,ij])
        
                    if ifit != []:
                        x_range = (ii, ii+1)
                        y_range = (ij, ij+1)
                        linemaps.set_map('height', ifit['lines_params'][:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude', ifit['lines_params'][:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity', ifit['velocity'],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm', ifit['lines_params'][:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('sigma', ifit['broadening'],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('flux', ifit['flux'],
                                         x_range=x_range, y_range=y_range)

                        linemaps.set_map('height-err', ifit['lines_params_err'][:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude-err', ifit['lines_params_err'][:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity-err', ifit['velocity_err'],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm-err', ifit['lines_params_err'][:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('sigma-err', ifit['broadening_err'],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('flux-err', ifit['flux_err'],
                                         x_range=x_range, y_range=y_range)

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

        if verbose:
            logging.info("Extracting integrated spectra", color=True)
        
        calibration_coeff_map = self.get_calibration_coeff_map()
        calibration_laser_map = self.get_calibration_laser_map()
                
        # Create parameters file
        paramsfile = list()
                    
        # extract regions
        integ_spectra = list()

        regions = self.get_mask_from_ds9_region_file(
            regions_file_path, integrate=False)
        
        for iregion in range(len(regions)):
            logging.info("Fitting region %d/%d"%(iregion, len(regions)))
            
            if not (self.params.wavelength_calibration):
                raise Exception('Not implemented')
            else:
                axis = self.params.base_axis.astype(float)

            spectrumf = self._extract_spectrum_from_region(
                regions[iregion],
                subtract_spectrum=subtract,
                silent=True)

            spectrum = spectrumf(axis)

            
            ifit = self._fit_lines_in_spectrum(
                spectrum, snr_guess=None)
            lines = gvar.mean(self.inputparams.allparams.pos_guess)
                
            if ifit != []:

                all_fit_results = list()
                print 'Velocity of the first line (km/s):', ifit['velocity_gvar'][0]
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
                fitted_vector = np.zeros_like(spectrum)
                fitted_models = list()
                
            if self.params.wavenumber:
                linesmodel = 'Cm1LinesModel'
            else:
                linesmodel = 'NmLinesModel'
                
            spectrum_header = (
                self._get_integrated_spectrum_header(
                    iregion))

            self.write_fits(
                self._get_integrated_spectrum_path(
                    iregion),
                spectrum, fits_header=spectrum_header,
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
                        'Line: {} ----'.format(
                            fit_results['line_name']))
                    for ikey in fit_results:
                        print '{}: {}'.format(
                            ikey, fit_results[ikey])

            if plot and ifit != []:

                import pylab as pl
                ax1 = pl.subplot(211)
                ax1.plot(axis, spectrum, c= '0.3',
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
                ax2.plot(axis, spectrum - fitted_vector, c= 'red',
                         ls='-', lw=1.5, label='residual')
                ax2.grid()
                ax2.legend()
                pl.show()

            integ_spectra.append(spectrum)
             
        return paramsfile

    def _fit_lines_in_spectrum(self, spectrum, snr_guess=None, **kwargs):
        """Raw function for spectrum fitting.

        .. note:: Need the InputParams class to be defined before call
        (see :py:meth:`~orcs.core.HDFCube._prepare_input_params`).

        :param spectrum: The spectrum to fit (1d vector).

        :param snr_guess: Guess on the SNR of the spectrum. Necessary
          to make a Bayesian fit (If unknown you can set it to 'auto'
          to try an automatic mode, two fits are made - one with a
          predefined SNR and the other with the SNR deduced from the
          first fit). If None a classical fit is made.

        :param kwargs: (Optional) Model parameters that must be
          changed in the InputParams instance.    
        """
        if not hasattr(self, 'inputparams'):
            self._print_error('Input params not defined')
        # check snr guess param
        auto_mode = False
        bad_snr_param = False
        if snr_guess is not None:
            if isinstance(snr_guess, str):
                if snr_guess.lower() == 'auto':
                    auto_mode = True
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
                
        if bad_snr_param:
            raise ValueError("snr_guess parameter not understood. It can be set to a float, 'auto' or None.")

        try:
            print 'SNR GUESS', snr_guess
            warnings.simplefilter('ignore')
            _fit = orb.fit._fit_lines_in_spectrum(
                spectrum, self.inputparams,
                fit_tol = self.fit_tol,
                compute_mcmc_error=False,
                snr_guess=snr_guess,
                **kwargs)
            warnings.simplefilter('default')
        
        except Exception, e:
            warnings.warn('Exception occured during fit: {}'.format(e))
            import traceback
            print traceback.format_exc()
                        
            return []

        if auto_mode:
            snr_guess = np.nanmax(spectrum) / np.nanstd(spectrum - _fit['fitted_vector'])
            return self._fit_lines_in_spectrum(spectrum, snr_guess=snr_guess, **kwargs)
        else:
            return _fit

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
            self.params.axis_corr,
            self.params.zpd_index,
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
            self._print_error('No calibration laser map given. Please redo the last step of the data reduction')

        if self.params.wavelength_calibration:
            calib_map = (np.ones((self.dimx, self.dimy), dtype=float)
                    * self.params.nm_laser * self.params.axis_corr)
        
        elif (calib_map.shape[0] != self.dimx):
            calib_map = orb.utils.image.interpolate_map(
                calib_map, self.dimx, self.dimy)

        # calibration correction
        if hasattr(self, 'sky_velocity_map'):
            ratio = 1 + (self.sky_velocity_map / orb.constants.LIGHT_VEL_KMS)
            calib_map /= ratio
        
        self.calibration_laser_map = calib_map
        self.reset_calibration_coeff_map()
        return self.calibration_laser_map

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

    def extract_spectrum_bin(self, x, y, b, mean_flux=False, **kwargs):
        """Extract a spectrum integrated over a binned region.
    
        :param x: X position of the bottom-left pixel
        
        :param y: Y position of the bottom-left pixel
        
        :param b: Binning. If 1, only the central pixel is extracted

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).
    
        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if b < 1: self._print_error('Binning must be at least 1')
        
        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[int(x):int(x+b), int(y):int(y+b)] = True
        region = np.nonzero(mask)

        return self.params.base_axis.astype(float), self._extract_spectrum_from_region(
            region, mean_flux=mean_flux, **kwargs)(self.params.base_axis.astype(float))


    def extract_spectrum(self, x, y, r, mean_flux=False, **kwargs):
        """Extract a spectrum integrated over a circular region of a
        given radius.

        :param x: X position of the center
        
        :param y: Y position of the center
        
        :param r: Radius. If 0, only the central pixel is extracted.
    
        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).
          
        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if r < 0: r = 0.001
        X, Y = np.mgrid[0:self.dimx, 0:self.dimy]
        R = np.sqrt(((X-x)**2 + (Y-y)**2))
        region = np.nonzero(R <= r)
        
        return self.params.base_axis.astype(float), self._extract_spectrum_from_region(
            region, mean_flux=mean_flux, **kwargs)(self.params.base_axis.astype(float))

    def extract_integrated_spectrum(self, region, mean_flux=False, **kwargs):
        """Extract a spectrum integrated over a given region (can be a
        list of pixels as returned by the function
        :py:meth:`numpy.nonzero` or a ds9 region file).

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param region: Region to integrate (can be a list of pixel
          coordinates as returned by the function
          :py:meth:`numpy.nonzero` or the path to a ds9 region
          file). If it is a ds9 region file, multiple regions can be
          defined and all will be integrated into one spectrum.
        """
        region = self.get_mask_from_ds9_region_file(region)
        return self.params.base_axis.astype(float), self._extract_spectrum_from_region(
            region, mean_flux=mean_flux, **kwargs)(self.params.base_axis.astype(float))



    def fit_lines_in_spectrum_bin(self, x, y, b, lines, nofilter=False,
                                  subtract_spectrum=None,
                                  snr_guess=None,
                                  mean_flux=False, 
                                  **kwargs):
        """Fit lines of a spectrum extracted from a circular region of a
        given radius.

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
          to make a Bayesian fit. If None a classical fit is made.    

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).
    
        :param kwargs: Keyword arguments of the function
          :py:meth:`orb.fit.fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`orb.fit.fit_lines_in_spectrum`)
        """

        axis, spectrum = self.extract_spectrum_bin(
            x, y, b, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux)
            
        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)
        
        fit_res = self._fit_lines_in_spectrum(
            spectrum, snr_guess=snr_guess)
                
        return axis, spectrum, fit_res

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
          to make a Bayesian fit. If None a classical fit is made.    

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`~HDFCube._fit_lines_in_spectrum`)
        """
        axis, spectrum = self.extract_spectrum(
            x, y, r, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux)

        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)
        
        fit_res = self._fit_lines_in_spectrum(
            spectrum, snr_guess=snr_guess)
                
        return axis, spectrum, fit_res

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
          to make a Bayesian fit. If None a classical fit is made.    

        :param mean_flux: (Optional) If True the flux of the spectrum
          is the mean flux of the extracted region (default False).

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.

        :returns: a tuple (axis, spectrum, fit_dict). fit_dict is a
          dictionary containing the fit results (same output as
          :py:meth:`~HDFCube._fit_lines_in_spectrum`)

        """
        axis, spectrum = self.extract_integrated_spectrum(
            region, subtract_spectrum=subtract_spectrum, mean_flux=mean_flux)

        self._prepare_input_params(lines, nofilter=nofilter, **kwargs)
        
        fit_res = self._fit_lines_in_spectrum(
            spectrum, snr_guess=snr_guess)
                
        return axis, spectrum, fit_res    

    def fit_lines_in_region(self, region, lines, binning=1, nofilter=False,
                            subtract_spectrum=None, **kwargs):
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

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._fit_lines_in_spectrum`.        
        """
        region = self.get_mask_from_ds9_region_file(region)
        self._prepare_input_params(
            lines, nofilter=nofilter, **kwargs)
        self._fit_lines_in_region(
            region, subtract_spectrum=subtract_spectrum,
            binning=binning)

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
                header=self.get_cube_header(),
                integrate=integrate)
        else: return region
        
    def correct_wavelength(self, sky_map_path):
        """Correct the wavelength of the cube based on the velocity of
        the sky lines computed with
        :py:meth:`~orcs.process.SpectralCube.map_sky_velocity`

        :param sky_map_path: Path to the sky velocity map.
        """
        sky_map = self.read_fits(sky_map_path)
        if sky_map.shape != (self.dimx, self.dimy):
            self._print_error('Given sky map does not have the right shape')

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
        if (dxmap.shape == (self.dimx, self.dimy)
            and dymap.shape == (self.dimx, self.dimy)):
            self.dxmap = dxmap
            self.dymap = dymap

    def pix2world(self, xy, deg=True):
        """Convert pixel coordinates to celestial coordinates

        :param xy: A tuple (x,y) of pixel coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...)

        :param deg: (Optional) If true, celestial coordinates are
          returned in sexagesimal format (default False).
        """
        xy = np.squeeze(xy)
        if np.size(xy) == 2:
            x = [xy[0]]
            y = [xy[1]]
        elif np.size(xy) > 2 and len(xy.shape) == 2:
            if xy.shape[1] > xy.shape[2]:
                xy = np.copy(xy.T)
            x = xy[:,0]
            y = xy[:,1]
            self._print_error('xy must be a tuple (x,y) of coordinates or a list of tuples ((x0,y0), (x1,y1), ...)')

        if not hasattr(self, 'dxmap') or not hasattr(self, 'dymap'):
            coords = np.array(
                self.wcs.all_pix2world(
                    x, y, 0)).T
        else:    
            coords = orb.utils.astrometry.pix2world(
                self.hdr, self.dimx, self.dimy, xy, self.dxmap, self.dymap)
        if deg: return coords
        else: return np.array(
            [orb.utils.astrometry.deg2ra(coords[:,0]),
             orb.utils.astrometry.deg2dec(coords[:,1])])


    def world2pix(self, radec, deg=True):
        """Convert celestial coordinates to pixel coordinates

        :param xy: A tuple (x,y) of celestial coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...). Must be in degrees.
        """
        radec = np.squeeze(radec)
        if np.size(radec) == 2:
            ra = [radec[0]]
            dec = [radec[1]]
        elif np.size(radec) > 2 and len(radec.shape) == 2:
            if radec.shape[1] > radec.shape[2]:
                radec = np.copy(radec.T)
            ra = radec[:,0]
            dec = radec[:,1]
            self._print_error('radec must be a tuple (ra,dec) of coordinates or a list of tuples ((ra0,dec0), (ra1,dec1), ...)')

        if not hasattr(self, 'dxmap') or not hasattr(self, 'dymap'):
            coords = np.array(
                self.wcs.all_world2pix(
                    ra, dec, 0,
                    detect_divergence=False,
                    quiet=True)).T
        else:    
            coords = orb.utils.astrometry.world2pix(
                self.hdr, self.dimx, self.dimy, radec, self.dxmap, self.dymap)

        return coords

    def get_deep_frame(self):
        """Return deep frame if if exists. None if no deep frame is
        attached to the cube."""
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'deep_frame' in f:
                return f['deep_frame'][:]
            else: return None
        
##################################################
#### CLASS LineMaps ##############################
##################################################

class LineMaps(orb.core.Tools):
    """Manage line parameters maps"""


    lineparams = ('height', 'height-err', 'amplitude', 'amplitude-err',
                  'velocity', 'velocity-err', 'fwhm', 'fwhm-err',
                  'sigma', 'sigma-err', 'flux', 'flux-err')


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
            self.dimx = int(int(dimx / self.div_nb) / self.binning) * self.div_nb
            self.dimy = int(int(dimy / self.div_nb) / self.binning) * self.div_nb
        else:
            self.dimx = dimx
            self.dimy = dimy

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
            self._print_error('Bad parameter')
         
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "MAPS" + os.sep
                + basename + "map.{}.{}x{}.{}.fits".format(
                    line_name, binning, binning, param))


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
        binnings = np.arange(self.binning+1, 50)
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

        if param not in self.lineparams:
            self._print_error('Bad parameter')
            
        if x_range is None and y_range is None:
            self.data[param] = data_map
        else:
            self.data[param][
                min(x_range):max(x_range),
                min(y_range):max(y_range)] = data_map

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
            
            if 'fwhm' in param:
                unit = ' [in {}]'.format(self.unit)
            elif 'velocity' in param:
                unit = ' [in km/s]'
            else: unit = ''
            
            for iline in range(len(self.lines)):
                line_name = self.line_names[iline]
                map_path = self._get_map_path(
                    line_name, param=param)

                new_map = self.data[param][:,:,iline]

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
