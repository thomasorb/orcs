#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orcs.py

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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
ORCS (Outils de Réduction de Cubes Spectraux) base processing module
provides tools to fix WCS and extract data from ORBS spectral cubes.

.. note:: ORCS is built over ORB so that ORB must be
  installed. Please make sure to give ORCS the correct path to ORB
  files
"""

import version
__version__ = version.__version__

# import Python libraries
import os
import sys
import math

import numpy as np
import pywcs
import bottleneck as bn

# import ORB
try:
    from orb.core import (Tools, OptionFile, Lines,
                          ProgressBar, ParamsFile, Cube)
    import orb.utils

except IOError, e:
    print "ORB could not be found !"
    print e
    sys.exit(2)

from rvcorrect import RVCorrect
import utils

##################################################
#### CLASS Orcs ##################################
##################################################

class Orcs(Tools):
    """ORCS user-interface.

    Manage the option file and the file system. This is the equivalent
    in ORCS of :py:meth:`orbs.Orbs` for ORBS.

    Most of the keywords used by ORBS are also used by ORCS. In
    general it is easier to include the option file used for the
    reduction of the data (using the keyword 'INCLUDE' followed by the
    path to the option file to include) and set the options which are
    special to ORCS. e.g.::

      INCLUDE /path/to/orbs.opt # include the ORBS option file
      
      DIRSPEC /path/to/CALIBRATED_SPECTRUM # Path to the spectrum cube folder
      LINES [NII]6548,Halpha,[NII]6583,[SII]6716,[SII]6731 # Lines to extract
      OBJECT_VELOCITY 170 # in km.s-1, mean velocity of the object
      POLY_ORDER 0 # Order of the polynomial used to fit continuum

   
    ORCS keywords (for the other keywords see :py:class:`orbs.orbs.Orbs`):

    :INCLUDE: Include another option file. Note that the already
      existing keywords will be overriden. This keyword is thus best
      placed at the very beginning of a file. All the keywords (but
      the 'protected ones') contained it the included file can be
      overriden by setting the same keyword after the include (see
      :py:class:`orb.core.OptionFile`).
    
    :DIRSPEC: Path to the folder containing the spectrum cube
    
    :LINES: Wavelength (in nm) of the lines to extract separated by a
      comma. Their full name can also be given but it must respect the list
      of recorded lines by :py:class:`orb.core.Lines`.
      
    :OBJECT_VELOCITY: (Optional) Mean velocity of the observed object
      in km.s-1. This is used to guess the mean shift of the lines.

    :POLY_ORDER: (Optional) Order of the polynomial used to fit
      continuum. Be extremely careful with high order polynomials.

    :ROI: (Optional) Region of interest Xmin,Xmax,Ymin,Ymax (in
      pixels). Must be intergers separated by a comma, e.g.:
      175,250,165,245

    :SKY_REG: (Optional) Path to a ds9 region file defining the sky pixels.
  
    :CALIBMAP: (Optional) Path to the calibration laser map. Must be
      given if the cube has not been wavelength calibrated.


    .. seealso:: :py:class:`orbs.orbs.Orbs`

    .. seealso:: :py:class:`orb.core.OptionFile`

    """

    config = dict()
    options = dict()
    overwrite = None
    optionfile = None
    spectralcube = None # SpectralCube instance
    wcs = None # pywcs.WCS Instance
    
    def __init__(self, option_file_path, config_file_name="config.orb",
                 overwrite=True):
        """Init Orcs class.

        :param option_file_path: Path to the option file.

        :param config_file_name: (Optional) name of the config file to
          use. Must be located in ``orb/data/``

        :param overwrite: (Optional) If True, any existing FITS file
          created by Orcs will be overwritten during the reduction
          process (default False).
        """
        def store_config_parameter(key, cast):
            if cast is not bool:
                self.config[key] = cast(
                    self._get_config_parameter(key))
            else:
                self.config[key] = bool(int(
                    self._get_config_parameter(key)))

        
        def store_option_parameter(option_key, key, cast, split=None,
                                   optional=False, folder=False,
                                   post_cast=str):
            value = self.optionfile.get(key, cast)
            
            if value is not None:
                if split is not None:
                    value = value.split(split)
                    value = np.array(value).astype(post_cast)
                if folder:
                    list_file_path =os.path.join(
                        self._get_project_dir(), key + ".list")
                    value = self._create_list_from_dir(
                        value, list_file_path)
                self.options[option_key] = value
            elif not optional:
                self._print_error(
                    "Keyword '{}' must be set".format(key))
                
                
        
        self.config_file_name = config_file_name
        self.overwrite = overwrite
        self.__version__ = version.__version__
        self._logfile_name =  os.path.basename(option_file_path) + '.log'

        ## get config parameters
        store_config_parameter('OBS_LAT', float)
        store_config_parameter('OBS_LON', float)
        store_config_parameter('OBS_ALT', float)
        store_config_parameter("CALIB_NM_LASER", float)

        ## get option file parameters
        
        # Observation parameters
        self.optionfile = OptionFile(option_file_path)
        store_option_parameter('object_name', 'OBJECT', str)
        store_option_parameter('filter_name', 'FILTER', str)
        store_option_parameter('apodization', 'APOD', float)
        self.options['project_name'] = (
            self.options['object_name']
            + '_' + self.options['filter_name']
            + '_' + str(self.options['apodization'])
            + '.ORCS')
        store_option_parameter('spectrum_list_path', 'DIRSPEC', str,
                               folder=True)
        
        store_option_parameter('step', 'SPESTEP', float)
        store_option_parameter('order', 'SPEORDR', int)
        store_option_parameter('step_nb', 'SPESTNB', int)

        # wavenumber
        self.options['wavenumber'] = False
        store_option_parameter('wavenumber', 'WAVENUMBER', bool,
                               optional=True)
        
        if self.options['wavenumber']:
            self._print_msg('Cube is in WAVENUMBER (cm-1)')
            unit = 'cm-1'
        else:
            self._print_msg('Cube is in WAVELENGTH (nm)')
            unit = 'nm'

        # wavelength calibration
        self.options['wavelength_calibration'] = True
        store_option_parameter('wavelength_calibration', 'WAVE_CALIB', bool,
                               optional=True)
        if self.options['wavelength_calibration']:
            self._print_msg('Cube is CALIBRATED in wavelength')
            self.options['calibration_laser_map_path'] = None
        else:
            self._print_msg('Cube is NOT CALIBRATED in wavelength')
            store_option_parameter('calibration_laser_map_path',
                                   'CALIBMAP', str)

        # target position
        store_option_parameter('target_ra', 'TARGETR', str, ':',
                               post_cast=float)
        store_option_parameter('target_dec', 'TARGETD', str, ':',
                               post_cast=float)
        store_option_parameter('target_x', 'TARGETX', float)
        store_option_parameter('target_y', 'TARGETY', float)
        store_option_parameter('obs_date', 'OBSDATE', str, '-',
                               post_cast=int)
        store_option_parameter('hour_ut', 'HOUR_UT', str, ':',
                               post_cast=float)

        # optional parameters
        store_option_parameter('roi', 'ROI', str, ',', post_cast=int,
                               optional=True)
        store_option_parameter('sky_regions_path', 'SKY_REG', str,
                               optional=True)
        
        ## Get lines parameters
        # Lines nm
        self.options['lines'] = self.optionfile.get_lines()
        self._print_msg('Searched lines (in nm): {}'.format(
            self.options['lines']))
        if self.options['wavenumber']:
            self.options['lines'] = orb.utils.nm2cm1(self.options['lines'])
            self._print_msg('Searched lines (in {}): {}'.format(
                unit, self.options['lines']))
        
        # Lines shift
        self.options['object_velocity'] = 0.
        store_option_parameter('object_velocity', 'OBJECT_VELOCITY', float,
                               optional=True)
        self._print_msg('Mean object velocity: {} km.s-1'.format(
            self.options['object_velocity']))
        
        # LSR
        self._print_msg('Radial velocity correction: {} km.s-1'.format(
           self.get_lsr_velocity()))

        # Total mean velocity
        self.options['mean_velocity'] = (
            self.options['object_velocity']
            + self.get_lsr_velocity())
        
        self._print_msg('Mean velocity shift: {} km.s-1, {:.3f} {}'.format(
            self.options['mean_velocity'], self.get_lines_shift(), unit))

        
        self._print_msg('Expected lines FWHM: {:.3f} {}'.format(
            self.get_lines_fwhm(), unit))
        
        # Continuum polynomial
        self.options['poly_order'] = 0
        store_option_parameter('poly_order', 'POLY_ORDER', int,
                               optional=True)
        self._print_msg(
            'Order of the polynomial used to fit continuum: {}'.format(
                self.options['poly_order']))

        # filter boundaries
        self.options['filter_range'] = orb.utils.read_filter_file(
            self._get_filter_file_path(self.options['filter_name']))[2:]
        if self.options['wavenumber']:
            self.options['filter_range'] = orb.utils.nm2cm1(
                self.options['filter_range'])

        ## Get WCS header
        cube = Cube(self.options['spectrum_list_path'],
                    silent_init=True)
        self.wcs = pywcs.WCS(cube.get_frame_header(0))
        
        ## Init spectral cube
        self.spectralcube = SpectralCube(
            self.options['spectrum_list_path'],
            data_prefix=self._get_data_prefix(),
            project_header=self._get_project_fits_header(),
            wcs_header = self.wcs.to_header(),
            overwrite=self.overwrite,
            logfile_name = self._logfile_name)

    def _get_project_dir(self):
        """Return the path to the project directory depending on 
        the project name."""
        return os.curdir + os.sep + self.options["project_name"] + os.sep

    def _get_data_prefix(self):
        """Return prefix path to stored data"""
        return (self._get_project_dir()
                + self.options['object_name']
                + '_' + self.options['filter_name']
                + '_' + str(self.options['apodization']) + '.')

    def _get_project_fits_header(self):
        """Return the header of the project that can be added to the
        created FITS files."""
        hdr = list()
        hdr.append(('ORCS', '{:s}'.format(self.__version__), 'ORCS version'))
        hdr += self.wcs.to_header()
        return hdr
    
    def get_lines_fwhm(self):
        """Return the expected value of the lines FWHM in nm or in
        cm-1 depending on the cube axis unit.  
          
        .. seealso:: :py:meth:`utils.compute_line_fwhm`
        """
        return utils.compute_line_fwhm(
            self.options['step_nb'], self.options['step'],
            self.options['order'], apod_coeff=self.options['apodization'],
            wavenumber=self.options['wavenumber'])

    def get_lsr_velocity(self):
        """Return LSR velocity in km.s-1

        .. seealso:: :py:class:`orcs.rvcorrect.RVCorrect` a
          translation in Python of the IRAF function RVCORRECT.
        """
        target_ra = self.options['target_ra']
        target_ra = ':'.join((str(int(target_ra[0])),
                              str(int(target_ra[1])),
                              str(target_ra[2])))
        
        target_dec = self.options['target_dec']
        target_dec = ':'.join((str(int(target_dec[0])),
                              str(int(target_dec[1])),
                              str(target_dec[2])))
        
        hour_ut = self.options['hour_ut']
        hour_ut = ':'.join((str(int(hour_ut[0])),
                            str(int(hour_ut[1])),
                            str(int(hour_ut[2]))))
        
        obs_coords = (self.config['OBS_LAT'],
                      self.config['OBS_LON'],
                      self.config['OBS_ALT'])
        
        return RVCorrect(target_ra, target_dec,
                         self.options['obs_date'], hour_ut,
                         obs_coords, silent=True).rvcorrect()[1]

    def get_lines_shift(self):
        """Return the expected line shift in nm or in cm-1 depending
        on the cube axis unit.

        .. seealso:: :py:meth:`utils.compute_line_shift`
        """
        return utils.compute_line_shift(
            self.options['mean_velocity'],
            self.options['step_nb'], self.options['step'],
            self.options['order'],
            wavenumber=self.options['wavenumber'])
            
    def extract_lines_maps(self):
        
        if self.options.get('roi') is not None:
            x_range = self.options['roi'][:2]
            y_range = self.options['roi'][2:]
        else:
            x_range = None
            y_range = None
            
        self.spectralcube.extract_lines_maps(
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            self.options['filter_range'],
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            poly_order=self.options.get('poly_order'),
            calibration_laser_map_path=self.options[
                'calibration_laser_map_path'],
            nm_laser = self.config['CALIB_NM_LASER'],
            sky_regions_file_path=self.options.get('sky_regions_path'),
            x_range=x_range,
            y_range=y_range)

#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCube(Cube):

    """ORCS spectral cube processing class.

    Fit and/or extract spectra.
    """

    def _get_maps_list_path(self):
        """Return path to the list of extacted emission lines maps"""
        return self._data_path_hdr + "maps_list"

    def _get_fits_header(self, file_type, comment=None):
        return (self._get_basic_header('Deep frame')
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))

    def _get_map_path(self, line_name, err=False, param="amplitude"):
        """Return the path to a map of one gaussian fit parameter for
        one given emission line.

        :param line_name: Name of the emission line

        :param param: (Optional) Parameter to map. Must be
          'amplitude', 'height', 'velocity', 'fwhm', 'chi' or 'snr'
          (Default 'amplitude').

        :param err: (Optional) True if the map is an error map (Defaut
          False).
          
        """
        if param == 'amplitude':
            param = 'AMP'
        elif param == 'height':
            param = 'HEI'
        elif param == 'velocity':
            param = 'VEL'
        elif param =='fwhm':
            param = 'FWHM'
        elif param =='snr':
            param = 'SNR'
        elif param =='chi':
            param = 'CHI'
        else:
            self._print_error("param must be 'amplitude', 'height', 'velocity', 'fwhm', 'chi' or 'snr'")
            
        if err:
            err = '_ERR'
        else:
            err = ''
            
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "MAPS" + os.sep
                + basename + "%s_MAP%s_%d.fits" %(param, err, line_name))

    def _get_integrated_spectra_fit_params_path(self):
        return self._data_path_hdr + "integ_spectra_fit_params"

    def _get_deep_frame_path(self):
        return self._data_path_hdr + "deep_frame.fits"

    def _get_raw_line_map_path(self, line_nm):
        return self._data_path_hdr + "%s_raw.fits"%str(line_nm)

    def _get_raw_err_map_path(self, line_nm):
        return self._data_path_hdr + "%s_raw_err.fits"%str(line_nm)

    def _get_sky_mask_path(self):
        return self._data_path_hdr + "sky_mask.fits"

    def _get_noise_map_path(self):
        return self._data_path_hdr + "noise_map.fits"
    
    def _get_integrated_spectrum_path(self, region_name):
        return self._data_path_hdr + "integrated_spectrum_%s"%region_name
    
    def _get_integrated_spectrum_map_path(self, region_name):
        return self._data_path_hdr + "integrated_spectrum_map_%s"%region_name


    def _get_fwhm_pix(self):
        """Return fwhm in channels from default lines FWHM"""
        return orb.utils.nm2pix(self.nm_axis,
                                self.nm_axis[0] + self.lines_fwhm)
    
    def _extract_spectrum_from_region(self, region,
                                      calibration_coeff_map,
                                      wavenumber, step, order,
                                      substract_spectrum=None,
                                      median=False, silent=False):
        """
        Extract the mean spectrum from a region of the cube.
        
        :param region: A list of the indices of the pixel integrated
          in the returned spectrum.

        :param calibration_coeff_map: Map of the calibration
          coefficient (calibration laser map / calibration laser
          wavelength). If the cube is wavelength calibrated this map
          must be a map of ones.

        :param wavenumber: If True cube is in wavenumber, else it is
          in wavelength.

        :param step: Step size in nm

        :param order: Folding order  
    
        :substract_spectrum: Remove the given spectrum from the
          extracted spectrum before fitting parameters. Useful
          to remove sky spectrum. Both spectra must have the same size.

        :param median: If True the integrated spectrum is the median
          of the spectra. Else the integrated spectrum is the mean of
          the spectra (Default False).

        :param silent: (Optional) If True, nothing is printed (default
          False).
        """
        def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
                                        median,
                                        wavenumber, base_axis, step, order):
            
            for icol in range(data_col.shape[0]):
                corr = calib_coeff_col[icol]
                if mask_col[icol]:
                    if wavenumber:
                        corr_axis = orb.utils.create_cm1_axis(
                            data_col.shape[1], step, order, corr=corr)
                        data_col[icol, :] = orb.utils.interpolate_axis(
                            data_col[icol, :], base_axis, 5,
                            old_axis=corr_axis)
                    else:
                        corr_axis = orb.utils.create_nm_axis_ireg(
                            data_col.shape[1], step, order, corr=1./corr)
                        data_col[icol, :] = orb.utils.interpolate_axis(
                            data_col[icol, :], base_axis, 5,
                            old_axis=corr_axis)
                else:
                    data_col[icol, :].fill(np.nan)
                    
            if median:
                return bn.nanmedian(data_col, axis=0)
            else:   
                return bn.nanmean(data_col, axis=0)
            
            

        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1

        spectrum = np.zeros(self.dimz, dtype=float)

        if np.any(calibration_coeff_map != 1.):
            if wavenumber:
                base_axis = orb.utils.create_cm1_axis(
                    self.dimz, step, order, corr=1.)
            else:
                base_axis  = orb.utils.create_nm_axis_ireg(
                    self.dimz, step, order, corr=1.)

            for iquad in range(0, self.QUAD_NB):
                x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
                iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                           0, self.dimz)

                # multi-processing server init
                job_server, ncpus = self._init_pp_server()
                if not silent: progress = ProgressBar(x_max - x_min)
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
                              median, wavenumber, base_axis, step, order), 
                        modules=('import bottleneck as bn',
                                 'import numpy as np',
                                 'import orb.utils'),
                        depfuncs=()))
                            for ijob in range(ncpus)]
                    for ijob, job in jobs:
                        result = job()
                        if not np.all(np.isnan(result)):
                            spectrum = (spectrum + result)/2.

                    if not silent:
                        progress.update(ii, info="column : {}/{}".format(
                            ii, self.dimx))
                self._close_pp_server(job_server)
                if not silent: progress.end()
            
        else: # faster if cube is calibrated
            if not silent: progress = ProgressBar(self.dimz)
            for iframe in range(self.dimz):
                spectra = (
                    np.array((self[:,:,iframe])[
                        np.nonzero(mask)].flatten()).astype(float))
                if median:
                    spectrum[iframe] = bn.nanmedian(spectra)
                else:
                    spectrum[iframe] = bn.nanmean(spectra)
                if not silent: progress.update(iframe)
            if not silent: progress.end()

        if substract_spectrum is not None:
            spectrum -= substract_spectrum
            
        return spectrum
        
    def _fit_lines_in_cube(self, lines, step, order, fwhm_guess,
                           wavenumber, calibration_coeff_map,
                           poly_order, filter_range,
                           x_range=None, y_range=None, fix_fwhm=False,
                           fmodel='gaussian', substract=None,
                           cov_pos=True, cov_fwhm=True):
        
        """Fit lines in a spectral cube.

        :param lines: searched lines. Must be given in the unit of the
          cube (cm-1 if wavenumber is True, nm if False)

        :param step: Step size in nm

        :param order: Folding order

        :param fwhm_guess: Expected lines FWHM. Must be given in the
          unit of the cube (cm-1 if wavenumber is True, nm if False)

        :param wavenumber: If True the cube is in wavenumber, else it
          is in wavelength.
          
        :param calibration_coeff_map: Map of the calibration
          coefficient (calibration laser map / calibration laser
          wavelength). If the cube is wavelength calibrated this map
          must be a map of ones.

        :param poly_order: Order of the polynomial used to fit
          continuum.

        :param filter_range: A tuple giving the min and max
          wavelength/wavenumber of the filter bandpass. Must be given
          in the unit of the cube (cm-1 if wavenumber is True, nm if
          False)

        :param x_range: (Optional) Range of pixels along the x axis
          where lines are fitted (Default None).
          
        :param y_range: (Optional) Range of pixels along the y axis
          where lines are fitted (Default None).

        :param fix_fwhm: (Optional) If True FWHM is fixed to
          fwhm_guess value (defautl False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param substract: (Optional) If not None this spectrum will be
          substracted before fitting.

        :param cov_pos: (Optional) Lines positions in each spectrum
          are shifted by the same value.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.
        """
        
        def _fit_lines_in_column(data, calib_map_col, mask_col,
                                 lines, fwhm_guess, filter_range,
                                 poly_order, fix_fwhm, fmodel,
                                 step, order, wavenumber,
                                 cov_pos, cov_fwhm, substract):
            RANGE_BORDER_PIX = 3

            fit = np.empty((data.shape[0], len(lines), 4),
                           dtype=float)
            fit.fill(np.nan)
            
            err = np.empty((data.shape[0], len(lines), 4),
                           dtype=float)
            err.fill(np.nan)
            
            res = np.empty((data.shape[0], len(lines), 2),
                           dtype=float)
            res.fill(np.nan)
            
            # convert lines wavelength/wavenumber into calibrated
            # positions
            if np.all(calib_map_col == 1.):
                if wavenumber:
                    cm1_axis = orb.utils.create_cm1_axis(
                        data.shape[1], step, order, corr=1.)
                    lines_pix = orb.utils.cm12pix(cm1_axis, lines)
                    filter_range_pix = orb.utils.cm12pix(
                        cm1_axis, filter_range)
                else:
                    nm_axis = orb.utils.create_nm_axis(
                        data.shape[1], step, order)
                    lines_pix = orb.utils.nm2pix(nm_axis, lines)
                    filter_range_pix = orb.utils.nm2pix(
                        nm_axis, filter_range)

                
            for ij in range(data.shape[0]):
                if mask_col[ij]:
                    if calib_map_col[ij] != 1.:
                        # convert lines wavelength/wavenumber into
                        # uncalibrated positions
                        if wavenumber:
                            cm1_axis = orb.utils.create_cm1_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij])
                            lines_pix = orb.utils.cm12pix(cm1_axis, lines)
                            filter_range_pix = orb.utils.cm12pix(
                                cm1_axis, filter_range)
                            # interpolate and substract
                            if substract is not None:
                                cm1_axis_base = orb.utils.create_cm1_axis(
                                    data.shape[1], step, order, corr=1.)
                                data[ij,:] -= orb.utils.interpolate_axis(
                                   substract, cm1_axis, 5,
                                   old_axis=cm1_axis_base)
                        else:
                            nm_axis = orb.utils.create_nm_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij])
                            lines_pix = orb.utils.nm2pix(nm_axis, lines)
                            filter_range_pix = orb.utils.nm2pix(
                                nm_axis, filter_range)
                            # interpolate and substract
                            if substract is not None:
                                nm_axis_base = orb.utils.create_nm_axis(
                                    data.shape[1], step, order, corr=1.)
                                data[ij,:] -= orb.utils.interpolate_axis(
                                   substract, nm_axis, 5,
                                   old_axis=nm_axis_base)
                                
                    # basic substraction if no interpolation of the
                    # spectrum has to be done
                    elif substract is not None:
                        data[ij,:] - substract

                    # get signal range
                    min_range = np.min(filter_range_pix)
                    max_range = np.max(filter_range_pix)
                    min_range += RANGE_BORDER_PIX
                    max_range -= RANGE_BORDER_PIX

                    ## FIT
                    result_fit = orb.utils.fit_lines_in_vector(
                        data[ij,:],
                        lines_pix,
                        fwhm_guess=fwhm_guess,
                        cont_guess=None,
                        fix_cont=False,
                        poly_order=poly_order,
                        cov_pos=cov_pos,
                        cov_fwhm=cov_fwhm,
                        fix_fwhm=fix_fwhm,
                        fmodel=fmodel,
                        observation_params=[step, order],
                        signal_range=[min_range, max_range],
                        return_fitted_vector=False,
                        wavenumber=wavenumber)
                    
                else: result_fit = []
                    
                for iline in range(len(lines)):
                    if result_fit != []:
                        fit[ij,iline,:] = result_fit[
                            'lines-params'][iline, :]
                        
                        # return the wavelength/wavenumber instead of
                        # the position in the vector
                        if wavenumber:
                            fit[ij,iline,2] = orb.utils.pix2cm1(
                                cm1_axis, fit[ij,iline,2])
                        else:
                            fit[ij,iline,2] = orb.utils.pix2nm(
                                nm_axis, fit[ij,iline,2])
                        
                        err[ij,iline,:] = result_fit[
                            'lines-params-err'][iline, :]
                        res[ij,iline,:] = [
                            result_fit['reduced-chi-square'],
                            result_fit['snr'][iline]]
                            
                    else:
                        fit[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN')]
                        err[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN')]
                        res[ij,iline,:] = [float('NaN'), float('NaN')]
            

            return fit, err, res
        
        ## Fit lines
        # Create arrays
        if np.size(lines) == 1:
            lines = [lines]
            
        hei_image = np.empty((self.dimx, self.dimy, len(lines)),
                              dtype=float)
        hei_image.fill(np.nan)
        hei_image_err = np.empty_like(hei_image)
        hei_image_err.fill(np.nan)
        
        amp_image = np.empty_like(hei_image)
        amp_image.fill(np.nan)
        amp_image_err = np.empty_like(hei_image)
        amp_image_err.fill(np.nan)
        
        shi_image = np.empty_like(hei_image)
        shi_image.fill(np.nan)
        shi_image_err = np.empty_like(hei_image)
        shi_image_err.fill(np.nan)
        
        wid_image = np.empty_like(hei_image)
        wid_image.fill(np.nan)
        wid_image_err = np.empty_like(hei_image)
        wid_image_err.fill(np.nan)
        
        chisq_image = np.empty_like(hei_image)
        chisq_image.fill(np.nan)
        snr_image = np.empty_like(hei_image)
        snr_image.fill(np.nan)
        
        # Defining computation mask
        if x_range is not None:
            x_min = np.min(x_range)
            x_max = np.max(x_range)
            if x_min < 0:
                self._print_warning("x min must be >= 0, x min set to 0")
                x_min == 0
            if x_max > self.dimx:
                self._print_warning("x max must be <= %d, x max set to %d"%(
                    self.dimx, self.dimx))
                x_max = self.dimx
        else:
            x_min = 0
            x_max = self.dimx
            
        if y_range is not None:
            y_min = np.min(y_range)
            y_max = np.max(y_range)
            if y_min < 0:
                self._print_warning("y min must be >= 0, y min set to 0")
                y_min == 0
            if y_max > self.dimy:
                self._print_warning("y max must be <= %d, y max set to %d"%(
                    self.dimy, self.dimy))
                y_max = self.dimy
        else:
            y_min = 0
            y_max = self.dimy
            
        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[x_min:x_max, y_min:y_max] = True

        for iquad in range(0, self.QUAD_NB):
            # note that x_min, x_max ... are redefined as the quadrant
            # boundaries
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                       0, self.dimz)

            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus):
                # no more jobs than columns
                if (ii + ncpus >= x_max - x_min): 
                    ncpus = x_max - x_min - ii
                
                # jobs creation
                jobs = [(ijob, job_server.submit(
                    _fit_lines_in_column,
                    args=(iquad_data[ii+ijob,:,:],
                          calibration_coeff_map[x_min + ii + ijob,
                                                y_min:y_max],
                          mask[x_min + ii + ijob, y_min:y_max],
                          lines, fwhm_guess, filter_range,
                          poly_order, fix_fwhm, fmodel,
                          step, order, wavenumber,
                          cov_pos, cov_fwhm, substract), 
                    modules=("import numpy as np", 
                             "import orb.utils",
                             "import orcs.utils as utils"),
                    depfuncs=(orb.utils.fit_lines_in_vector,)))
                        for ijob in range(ncpus)]
            
                for ijob, job in jobs:
                    (fit, err, chi) = job()
                    hei_image[x_min+ii+ijob,y_min:y_max,:] = fit[:,:,0]
                    amp_image[x_min+ii+ijob,y_min:y_max,:] = fit[:,:,1]
                    shi_image[x_min+ii+ijob,y_min:y_max,:] = fit[:,:,2]
                    wid_image[x_min+ii+ijob,y_min:y_max,:] = fit[:,:,3]
                    hei_image_err[x_min+ii+ijob,
                                  y_min:y_max,:] = err[:,:,0]
                    amp_image_err[x_min+ii+ijob,
                                  y_min:y_max,:] = err[:,:,1]
                    shi_image_err[x_min+ii+ijob,
                                  y_min:y_max,:] = err[:,:,2]
                    wid_image_err[x_min+ii+ijob,
                                  y_min:y_max,:] = err[:,:,3]
                    chisq_image[x_min+ii+ijob,
                                y_min:y_max,:] = chi[:,:,0]
                    snr_image[x_min+ii+ijob,
                              y_min:y_max,:] = chi[:,:,1]
                
                progress.update(ii, info="column : {}/{}".format(
                    ii, self.dimx))
                
            self._close_pp_server(job_server)
            progress.end()
    
        results_list = list()
        for iline in range(len(lines)):
            results_list.append((hei_image[:,:,iline],
                                 amp_image[:,:,iline],
                                 shi_image[:,:,iline],
                                 wid_image[:,:,iline],
                                 hei_image_err[:,:,iline],
                                 amp_image_err[:,:,iline],
                                 shi_image_err[:,:,iline],
                                 wid_image_err[:,:,iline],
                                 chisq_image[:,:,iline],
                                 snr_image[:,:,iline]))
        return results_list


    ## def create_sky_mask(self, sky_regions_file_path, threshold_coeff=0.3):
    ##     """Create a mask from the pixels corresponding to the sky
        
    ##     :param sky_regions_file_path: Path to a file containing the
    ##       coordinates of the regions of the sky.

    ##     .. note:: The region file can be created using ds9 and the
    ##       region shape 'box'. The region file is then saved
    ##       ('region/save regions') using the default parameters
    ##       ('format : ds9', 'Coordinates system : physical').
    ##     """
    ##     sky_regions = orb.utils.get_mask_from_ds9_region_file(
    ##         sky_regions_file_path,
    ##         x_range=[0, self.dimx],
    ##         y_range=[0, self.dimy])
    ##     deep_frame = self.create_deep_frame(write=False)
    ##     sky = deep_frame[sky_regions]
    ##     # std cut to remove stars from the sky
    ##     sky_median = np.median(sky)
    ##     sky_std = np.std(sky)
    ##     sky[np.nonzero(sky > sky_median + sky_std)] = 0.
    ##     sky[np.nonzero(sky < 0)] = 0.
    ##     sky_median = np.median(sky[np.nonzero(sky)])
    ##     sky_std = np.std(sky[np.nonzero(sky)])
        
    ##     mask_map = np.ones_like(deep_frame)
    ##     mask_map[np.nonzero(
    ##         deep_frame < sky_median + threshold_coeff * sky_std)] = 0

    ##     self.write_fits(self._get_sky_mask_path(), mask_map,
    ##                     overwrite=self.overwrite,
    ##                     fits_header=self._get_fits_header("Sky Mask"))
    ##     return mask_map


    ## def get_sky_radial_velocity(self, sky_region_file_path, fmodel='gaussian',
    ##                             lines_shift=0., show=True):
    ##     """
    ##     .. note:: Use :py:class:`orb.core.Lines` to get the sky lines
    ##       to fit.

    
    
    ##     """
    ##     # compute signal range
    ##     signal_range = self._get_signal_range().astype(int)
        
    ##     sky_lines_nm = Lines().get_sky_lines(self.filter_min, self.filter_max,
    ##                                          self.lines_fwhm)
        
    ##     sky_lines_nm_shift = np.array(sky_lines_nm) + lines_shift
    ##     sky_lines_pix = orb.utils.nm2pix(self.nm_axis, sky_lines_nm_shift)
        
   
    ##     region = orb.utils.get_mask_from_ds9_region_file(
    ##         sky_region_file_path,
    ##         x_range=[0, self.dimx], y_range=[0, self.dimy])
    ##     sky_median_spectrum = self._extract_spectrum_from_region(
    ##         region, median=True)

    ##     cont_guess = orb.utils.robust_median(sky_median_spectrum)
    ##     fix_cont = False
        

    ##     fit = orb.utils.fit_lines_in_vector(
    ##         sky_median_spectrum,
    ##         np.copy(sky_lines_pix),
    ##         cont_guess=[cont_guess],
    ##         fix_cont=fix_cont,
    ##         poly_order=0,
    ##         fmodel=fmodel,
    ##         cov_pos=True,
    ##         cov_fwhm=True,
    ##         return_fitted_vector=True,
    ##         fwhm_guess=self._get_fwhm_pix(),
    ##         fix_fwhm=False,
    ##         signal_range=signal_range,
    ##         interpolation_params=[float(self.hdr['STEP']),
    ##                               float(self.hdr['ORDER'])])
    
    ##     fit_params = fit['lines-params']
    ##     fit_params_err = fit['lines-params-err']
        
            
    ##     velocity = list()
    ##     for iline in range(len(fit_params)):
    ##         pline = fit_params[iline]
    ##         velocity.append(utils.compute_radial_velocity(
    ##             orb.utils.pix2nm(self.nm_axis, pline[2]),
    ##             sky_lines_nm[iline]))
            
    ##         err = utils.compute_radial_velocity(
    ##             orb.utils.pix2nm(self.nm_axis, fit_params_err[iline][2]
    ##              + sky_lines_pix[iline]),
    ##             orb.utils.pix2nm(self.nm_axis, sky_lines_pix[iline]))
            
    ##         self._print_msg("Line %f nm, velocity : %f km/s [+/-] %f"%(
    ##             sky_lines_nm[iline], velocity[-1], err))
            
    ##     self._print_msg(
    ##         "Mean sky radial velocity : %f km/s"%np.mean(np.array(velocity)))

    ##     if show:
    ##         import pylab as pl
    ##         for iline in range(np.size(sky_lines_nm_shift)):
    ##             pl.axvline(x=sky_lines_nm_shift[iline], alpha=0.5,
    ##                        color='0.', linestyle=':')
    ##         pl.plot(self.nm_axis, sky_median_spectrum, color='0.5',
    ##                 label = 'sky spectrum')
    ##         pl.plot(self.nm_axis, fit['fitted-vector'], color='0.',
    ##                 linewidth=2., label='fit')
    ##         pl.legend()
    ##         pl.show()

    ##     err_params = np.nan
    ##     return sky_median_spectrum, fit_params, err_params
        
            
    def extract_lines_maps(self, lines, lines_velocity,
                           lines_fwhm, filter_range,
                           wavenumber, step, order,
                           poly_order=0,
                           calibration_laser_map_path=None,
                           nm_laser=None,
                           sky_regions_file_path=None,
                           cov_pos=True, cov_fwhm=True,
                           fix_fwhm=False,
                           fmodel='gaussian',
                           x_range=None, y_range=None):
        
        """
        Extract emission lines parameters maps from a fit.

        All parameters of each emission line are mapped. So that for a
        gaussian fit with 4 parameters (height, amplitude, shift,
        fwhm) 4 maps are created for each fitted emission line in the
        cube.

        :param lines: Lines rest frame. Must be given in the unit of
          the cube (cm-1 if wavenumber is True, nm if False)

        :param lines_velocity: Mean Lines velocity in km.s-1.

        :param lines_fwhm: Expected FWHM of the lines. Must be given
          in the unit of the cube (cm-1 if wavenumber is True, nm if
          False)

        :param filter_range: A tuple giving the min and max
          wavelength/wavenumber of the filter bandpass. Must be given
          in the unit of the cube (cm-1 if wavenumber is True, nm if
          False)

        :param wavenumber: If True the cube is in wavenumber else it
          is in wavelength.

        :param step: Step size in nm

        :param order: Folding order

        :param poly_order: Order of the polynomial used to fit
          continuum.

        :param calibration_laser_map_path: (Optional) If not None the
          cube is considered to be uncalibrated in wavelength. In this
          case the calibration laser wavelength (nm_laser) must also
          be given (default None).

        :param nm_laser: (Optional) Calibration laser wavelentgh. Must
          be given if calibration_laser_map_path is not None (default
          None).

        :sky_regions_file_path: (Optional) Path to a ds9 region file
          giving the pixels where the sky spectrum has to be
          extracted. This spectrum will be substracted to the spectral
          cube before it is fitted (default None).

        :param fix_fwhm: If True fix FWHM to its guess (default
          False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param cov_pos: (Optional) Lines positions in each spectrum
          are shifted by the same value.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.
        
        :param x_range: (Optional) Range of columns over which the
          extraction must be done.

        :param y_range: (Optional) Range of rows over which the
          extraction must be done.
        """
        self._print_msg("Extracting lines data", color=True)

        rest_frame_lines = np.copy(lines)
        lines += orb.utils.line_shift(lines_velocity, lines,
                                      wavenumber=wavenumber)

        if calibration_laser_map_path is None:
            calibration_laser_map = np.ones((self.dimx, self.dimy),
                                            dtype=float)
            nm_laser = 1.
            calibration_coeff_map = np.copy(calibration_laser_map)
        else:
            calibration_laser_map = self.read_fits(
                calibration_laser_map_path)
            if (calibration_laser_map.shape[0] != self.dimx):
                calibration_laser_map = orb.utils.interpolate_map(
                    calibration_laser_map, self.dimx, self.dimy)
            if nm_laser is None:
                self._print_error('If a calibration laser map path is given the wavelength of the calibration laser used must also be given. Please set nm_laser option.')
            calibration_coeff_map = calibration_laser_map / nm_laser
                
        
        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            self._print_msg("Extracting sky median vector")
            median_sky_spectrum = self._extract_spectrum_from_region(
                orb.utils.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy]),
                calibration_coeff_map,
                wavenumber, step, order)
        else:
            median_sky_spectrum = np.zeros(self.dimz, dtype=float)

        ## FIT ###
        self._print_msg("Fitting data")
        results = self._fit_lines_in_cube(
            lines, step, order, lines_fwhm,
            wavenumber, calibration_coeff_map,
            poly_order, filter_range,
            x_range=x_range,
            y_range=y_range,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            cov_pos=cov_pos, cov_fwhm=cov_fwhm,
            substract=median_sky_spectrum)

        ## SAVE MAPS ###
        maps_list_path = self._get_maps_list_path()
        maps_list = self.open_file(maps_list_path, 'w')

        if wavenumber:
            axis = orb.utils.create_cm1_axis(self.dimz, step, order)
        else:
            axis = orb.utils.create_nm_axis(self.dimz, step, order)
    
        delta = abs(axis[1] - axis[0])

        for iline in range(np.size(lines)):
            if wavenumber:
                line_name = Lines().round_nm2ang(
                    orb.utils.cm12nm(rest_frame_lines[iline]))
                unit = 'cm-1'
            else:
                line_name = Lines().round_nm2ang(rest_frame_lines[iline])
                unit = 'nm'
            
            # write height map
            map_path = self._get_map_path(line_name,
                                          param='height')
            self.write_fits(
                map_path, results[iline][0],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Height Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'HEI'))

            # write height map err
            map_path = self._get_map_path(line_name,
                                          param='height',
                                          err=True)
            self.write_fits(
                map_path, results[iline][4],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Height Map Error %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'HEI_ERR'))
            
            # write amp map
            map_path = self._get_map_path(line_name,
                                          param='amplitude')
            self.write_fits(
                map_path, results[iline][1],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Amplitude Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'AMP'))

            # write amp map err
            map_path = self._get_map_path(line_name,
                                          param='amplitude',
                                          err=True)
            self.write_fits(
                map_path, results[iline][5],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Amplitude Map Error%d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'AMP_ERR'))
            
            # write FWHM map
            map_path = self._get_map_path(line_name,
                                          param='fwhm')
            fwhm_map = results[iline][3] * delta
            self.write_fits(
                map_path, fwhm_map,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "FWHM Map %d"%line_name,
                    comment='in {}'.format(unit)))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'FWHM'))

            # write FWHM map err
            map_path = self._get_map_path(line_name,
                                          param='fwhm',
                                          err=True)
            fwhm_map = results[iline][7] * delta
            self.write_fits(
                map_path, fwhm_map,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "FWHM Map Error %d"%line_name,
                    comment='in {}'.format(unit)))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'FWHM_ERR'))


            # write velocity map
                
            velocity = utils.compute_radial_velocity(
                results[iline][2], rest_frame_lines[iline],
                wavenumber=wavenumber)
  
            map_path = self._get_map_path(line_name,
                                          param='velocity')
            self.write_fits(
                map_path, velocity,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Velocity Map %d"%line_name,
                    comment='in km/s'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'VEL'))

            # write velocity map err
            # compute radial velocity from shift

            velocity = np.abs(utils.compute_radial_velocity(
                rest_frame_lines[iline] + results[iline][6],
                rest_frame_lines[iline],
                wavenumber=wavenumber))
           
            map_path = self._get_map_path(line_name,
                                          param='velocity',
                                          err=True)
            self.write_fits(
                map_path, velocity,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Velocity Map Error %d"%line_name,
                    comment='in km/s'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'VEL_ERR'))

            # write SNR map
            map_path = self._get_map_path(line_name,
                                          param='snr')
            self.write_fits(
                map_path, results[iline][9],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "SNR Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'SNR'))

            # write chi-square map
            map_path = self._get_map_path(line_name,
                                          param='chi')
            self.write_fits(
                map_path, results[iline][8],
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "Reduced chi-square Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'CHI'))
                

    ## def extract_integrated_spectra(self, regions_file_path,
    ##                                sky_regions_file_path=None,
    ##                                median=False, poly_order=None,
    ##                                observation_params=None,
    ##                                signal_range=None, plot=True,
    ##                                auto_sky_extraction=True,
    ##                                sky_size_coeff=2.):
        
    ##     if poly_order is None:
    ##         poly_order = self.poly_order


    ##     # Create parameters file
    ##     paramsfile = orb.core.ParamsFile(
    ##         self._get_integrated_spectra_fit_params_path())
        
    ##     # Extract sky spectrum
    ##     if sky_regions_file_path is not None:
    ##         self._print_msg("Extracting sky median vector")
    ##         reg_mask = orb.utils.get_mask_from_ds9_region_file(
    ##                 sky_regions_file_path,
    ##                 x_range=[0, self.dimx],
    ##                 y_range=[0, self.dimy])

    ##         median_sky_spectrum = self._extract_spectrum_from_region(
    ##             reg_mask)
    ##     else: median_sky_spectrum = None
            

    ##     # extract regions
    ##     integ_spectra = list()
    ##     with open(regions_file_path, 'r') as f:
    ##         region_index = 0
    ##         for ireg in f:
    ##             if len(ireg) > 3 and '#' not in ireg:
    ##                 region = orb.utils.get_mask_from_ds9_region_line(
    ##                     ireg,
    ##                     x_range=[0, self.dimx],
    ##                     y_range=[0, self.dimy])
    ##                 if len(region[0]) > 0:
    ##                     if auto_sky_extraction:
    ##                         # get sky around region
    ##                         reg_bary = (np.mean(region[0]),
    ##                                     np.mean(region[1]))
    ##                         reg_size = max(np.max(region[0])
    ##                                        - np.min(region[0]),
    ##                                        np.max(region[1])
    ##                                        - np.min(region[1]))

    ##                         sky_size = reg_size * sky_size_coeff
    ##                         x_min, x_max, y_min, y_max = (
    ##                             orb.utils.get_box_coords(
    ##                                 reg_bary[0], reg_bary[1], sky_size,
    ##                                 0, self.dimx, 0, self.dimy))

    ##                         x_sky = list()
    ##                         y_sky = list()
    ##                         for ipix in range(int(x_min), int(x_max)):
    ##                             for jpix in range(int(y_min), int(y_max)):
    ##                                 if ((math.sqrt((ipix - reg_bary[0])**2.
    ##                                                + (jpix - reg_bary[1])**2.)
    ##                                      <= round(sky_size / 2.))):
    ##                                     x_sky.append(ipix)
    ##                                     y_sky.append(jpix)

    ##                         sky = list([x_sky, y_sky])
    ##                         mask = np.zeros((self.dimx, self.dimy))
    ##                         mask[sky] = 1
    ##                         mask[region] = 0
    ##                         sky = np.nonzero(mask)
                            
    ##                         median_sky_spectrum = (
    ##                             self._extract_spectrum_from_region(
    ##                                 sky, median=True, silent=True))
                            
    ##                     # extract spectrum
    ##                     spectrum = self._extract_spectrum_from_region(
    ##                         region, substract_spectrum=median_sky_spectrum,
    ##                         median=median, silent=True)

    ##                     pix_center = int(self.nm_axis.shape[0]/2.)
    ##                     shift_guess = (
    ##                         orb.utils.nm2pix(
    ##                             self.nm_axis,
    ##                             self.nm_axis[pix_center] + self.lines_shift)
    ##                         - pix_center)

    ##                     # fit spectrum
    ##                     fit = orb.utils.fit_lines_in_vector(
    ##                         spectrum,
    ##                         orb.utils.nm2pix(self.nm_axis, self.lines_nm),
    ##                         fwhm_guess=self._get_fwhm_pix(),
    ##                         cont_guess=None,
    ##                         shift_guess=shift_guess,
    ##                         fix_cont=False,
    ##                         poly_order=poly_order,
    ##                         cov_pos=True,
    ##                         cov_fwhm=True,
    ##                         fix_fwhm=False,
    ##                         fmodel='gaussian',
    ##                         interpolation_params=observation_params,
    ##                         signal_range=self._get_signal_range(),
    ##                         return_fitted_vector=True)

    ##                     if fit != []:
    ##                         fit_params = fit['lines-params']
    ##                         err_params = fit['lines-params-err']
    ##                         real_lines_nm = orb.utils.pix2nm(
    ##                             self.nm_axis, fit_params[:, 2])

    ##                         velocities = utils.compute_radial_velocity(
    ##                             real_lines_nm,
    ##                             self.lines_nm)

    ##                         err_lines_nm = orb.utils.pix2nm(
    ##                             self.nm_axis, err_params[:, 2])

    ##                         velocities_err = utils.compute_radial_velocity(
    ##                             err_lines_nm, self.nm_axis[0])
                            
    ##                         for iline in range(fit_params.shape[0]):
    ##                             snr = abs(fit_params[iline, 1]
    ##                                       / err_params[iline, 1])
    ##                             fit_results = {
    ##                                 'reg_index': region_index,
    ##                                 'line_name': self.lines.get_line_name(
    ##                                     self.lines_nm[iline]),
    ##                                 'h': fit_params[iline, 0],
    ##                                 'a': fit_params[iline, 1],
    ##                                 'v': velocities[iline],
    ##                                 'fwhm': fit_params[iline, 3],
    ##                                 'h_err': err_params[iline, 0],
    ##                                 'a_err': err_params[iline, 1],
    ##                                 'v_err': velocities_err[iline],
    ##                                 'fwhm_err': err_params[iline, 3],
    ##                                 'snr': snr}

    ##                             paramsfile.append(fit_results)
    ##                             print paramsfile[-1]

    ##                     if plot:
    ##                         import pylab as pl
    ##                         pl.plot(self.nm_axis, spectrum,
    ##                                 label='orig spectrum')
    ##                         if fit != []:
    ##                             pl.plot(self.nm_axis, fit['fitted-vector'],
    ##                                     label='fit')
    ##                         pl.legend()
    ##                         pl.show()
                            
    ##                     integ_spectra.append(spectrum)
                        
    ##             region_index += 1
                    
    ##     return integ_spectra
        

    ## def get_fitted_cube(self, x_range=None, y_range=None):

    ##     self._print_error('Not implemented yet')

    ##     self._print_msg("Extracting lines data", color=True)
            
    ##     #rest_frame_lines_nm = np.copy(self.lines_nm)
    ##     lines_nm = self.lines_nm + self.lines_shift
        
    ##     if x_range is None:
    ##         x_range = self.x_range
    ##     if y_range is None:
    ##         y_range = self.y_range
            
    ##     results = self._fit_lines_in_cube(
    ##         self.data,
    ##         searched_lines = orb.utils.nm2pix(
    ##             self.nm_axis, lines_nm),
    ##         signal_range=[orb.utils.nm2pix(self.nm_axis,
    ##                                        self.filter_min),
    ##                       orb.utils.nm2pix(self.nm_axis,
    ##                                        self.filter_max)],
    ##         x_range=x_range,
    ##         y_range=y_range,
    ##         fwhm_guess=int(self._get_fwhm_pix()),
    ##         detect_coeff=0.,
    ##         fix_cont=False)

    ##     x_min = np.min(x_range)
    ##     x_max = np.max(x_range)
    ##     y_min = np.min(y_range)
    ##     y_max = np.max(y_range)

    ##     fitted_cube = np.empty((x_max - x_min, y_max - y_min, self.dimz),
    ##                            dtype=float)
        
    ##     for ii in range(x_max - x_min):
    ##         for ij in range(y_max - y_min):
    ##             fitted_spectrum = np.zeros(self.dimz, dtype=float)
    ##             for iline in range(len(results)):
    ##                 amp = results[iline][1][ii + x_min,ij + y_min]
    ##                 dx = results[iline][2][ii + x_min,ij + y_min]
    ##                 fwhm = results[iline][3][ii + x_min,ij + y_min]
                    
    ##                 ## fit_min = int(dx - 3 * fwhm)
    ##                 ## fit_max = int(dx + 3 * fwhm + 1)
    ##                 ## if fit_min < 0: fit_min = 0
    ##                 ## if fit_max >= self.dimz: fit_max = self.dimz - 1
    ##                 gauss_line = orb.utils.gaussian1d(
    ##                     np.arange(self.dimz),
    ##                     0., amp, dx, fwhm)
    ##                 fitted_spectrum += gauss_line
                    
    ##             fitted_cube[ii,ij,:] = fitted_spectrum
                    
    ##     return fitted_cube

    ## def extract_raw_lines_maps(self):
    ##     """Raw extraction of a portion of the spectral cube. Return
    ##     the median of the slice.

    ##     :param line_nm: Wavelength in nm of the line to extract. Can
    ##       be a float or a string recognized by ORB (see
    ##       :py:class:`orb.core.Lines`)

    ##     .. note: Shift in the option file is taken into account
    ##     """
    ##     lines_nm = self.lines_nm + self.lines_shift
    ##     lines_pix = orb.utils.nm2pix(self.nm_axis, lines_nm)
        
    ##     fwhm_pix = self._get_fwhm_pix()
    ##     min_pix = np.floor(lines_pix - fwhm_pix)
    ##     min_pix[np.nonzero(min_pix < 0)] = 0
    ##     max_pix = np.ceil(lines_pix + fwhm_pix) + 1
    ##     max_pix[np.nonzero(max_pix >= self.dimz)] = self.dimz - 1

    ##     # get continuum map
    ##     mask_vector = np.ones(self.dimz, dtype=float)
    ##     for iline in range(len(lines_pix)):
    ##         mask_vector[min_pix[iline]:max_pix[iline]] = 0

    ##     cont_map = np.nanmean(np.squeeze(
    ##         self.data[:,:,np.nonzero(mask_vector)]), axis=2)

    ##     cont_map_std = np.nanstd(np.squeeze(
    ##         self.data[:,:,np.nonzero(mask_vector)]), axis=2)
        
    ##     # write maps
    ##     for iline in range(len(lines_pix)):
    ##         line_map = (bn.nanmean(
    ##             self.data[:,:,min_pix[iline]:max_pix[iline]], axis=2)
    ##                     - cont_map)
    ##         err_map = cont_map_std / math.sqrt(max_pix[iline] - min_pix[iline])
    
    ##         raw_path = self._get_raw_line_map_path(
    ##             int(round(self.lines_nm[iline]*10.)))
    ##         raw_header = self._get_fits_header(
    ##                 'RAW LINE MAP {:d}'.format(int(round(
    ##                 self.lines_nm[iline]*10.))))

    ##         err_path = self._get_raw_err_map_path(
    ##             int(round(self.lines_nm[iline]*10.)))
    ##         err_header = self._get_fits_header(
    ##                 'RAW ERR MAP {:d}'.format(int(round(
    ##                 self.lines_nm[iline]*10.))))
            
    ##         self.write_fits(raw_path, line_map, overwrite=self.overwrite,
    ##             fits_header=raw_header)
    ##         self.write_fits(err_path, err_map, overwrite=self.overwrite,
    ##             fits_header=err_header)
        
        
        
        
