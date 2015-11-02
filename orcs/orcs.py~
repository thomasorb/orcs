#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orcs.py

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
ORCS (Outils de Réduction de Cubes Spectraux) provides tools to
extract data from ORBS spectral cubes.

.. note:: ORCS is built over ORB so that ORB must be installed.
"""

import version
__version__ = version.__version__

# import Python libraries
import os
import sys
import math

import numpy as np
import astropy.wcs as pywcs
import astropy.io.fits as pyfits
import bottleneck as bn
import warnings

# import ORB
try:
    from orb.core import (
        Tools, OptionFile, Lines,
        ProgressBar, HDFCube)
    import orb.utils.spectrum
    import orb.utils.image
    import orb.utils.stats
    import orb.utils.filters
    

except IOError, e:
    print "ORB could not be found !"
    print e
    sys.exit(2)

from rvcorrect import RVCorrect

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
      
      CUBEPATH /path/to/CALIBRATED_SPECTRUM # Path to the spectrum cube (hdf5 format)
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
    
    :CUBEPATH: Path to the spectrum cube (hdf5 format)
    
    :LINES: Wavelength (in nm) of the lines to extract separated by a
      comma. Their full name can also be given but it must respect the
      list of recorded lines by :py:class:`orb.core.Lines`. A line
      wavelentgh in nm can also be given.
      
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
    wcs_header = None
    
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
        store_option_parameter('spectrum_cube_path', 'CUBEPATH', str)
        
        store_option_parameter('step', 'SPESTEP', float)
        store_option_parameter('order', 'SPEORDR', int)

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
            self._print_msg('Calibration laser map used: {}'.format(
                self.options['calibration_laser_map_path']))
            
        # WCS
        self.options['wcs'] = True
        store_option_parameter('wcs', 'WCS', bool, optional=True)
        if not self.options['wcs']:
            wcs_optional = True
        else:
            wcs_optional = False

        ## Get WCS header
        cube = HDFCube(self.options['spectrum_cube_path'],
                       silent_init=True)
        
        if self.options['wcs']:
            self.wcs = pywcs.WCS(cube.get_frame_header(0))
            self.wcs_header = self.wcs.to_header()

        # get ZPD index
        self.options['step_nb'] = cube.dimz
        if 'ZPDINDEX' in cube.get_cube_header():
            self.zpd_index = cube.get_cube_header()['ZPDINDEX']
        else:
            self._print_error('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')

        
        store_option_parameter('target_ra', 'TARGETR', str, ':',
                               post_cast=float, optional=wcs_optional)
        store_option_parameter('target_dec', 'TARGETD', str, ':',
                               post_cast=float, optional=wcs_optional)
        store_option_parameter('target_x', 'TARGETX', float, optional=wcs_optional)
        store_option_parameter('target_y', 'TARGETY', float, optional=wcs_optional)
        store_option_parameter('obs_date', 'OBSDATE', str, '-',
                               post_cast=int, optional=wcs_optional)
        store_option_parameter('hour_ut', 'HOUR_UT', str, ':',
                               post_cast=float, optional=wcs_optional)

        # optional parameters
        store_option_parameter('roi', 'ROI', str, ',', post_cast=int,
                               optional=True)
        store_option_parameter('sky_regions_path', 'SKY_REG', str,
                               optional=True)


        # Integrated spectra
        store_option_parameter('integ_reg_path', 'INTEG_REG_PATH', str,
                               optional=True)
        
        self.options['auto_sky_extraction'] = False
        store_option_parameter('auto_sky_extraction', 'INTEG_AUTO_SKY', bool,
                               optional=True)
        
        self.options['sky_size_coeff'] = 2.
        store_option_parameter('sky_size_coeff', 'INTEG_SKY_SIZE_COEFF', float,
                               optional=True)

        self.options['plot'] = True
        store_option_parameter('plot', 'INTEG_PLOT', bool,
                               optional=True)

        # filter boundaries
        self.options['filter_range'] = orb.utils.filters.read_filter_file(
            self._get_filter_file_path(self.options['filter_name']))[2:]
        if self.options['wavenumber']:
            self.options['filter_range'] = orb.utils.spectrum.nm2cm1(
                self.options['filter_range'])

        # print some info about the cube parameters
        self._print_msg('Step size: {} nm'.format(
            self.options['step']))
        
        self._print_msg('Folding order: {}'.format(
            self.options['order']))

        self._print_msg('Step number: {}'.format(
            self.options['step_nb']))
        
        ## Get lines parameters
        # Lines nm
        if self.options['wavenumber']:
            nm_min = orb.utils.spectrum.cm12nm(
                np.max(self.options['filter_range']))
            nm_max = orb.utils.spectrum.cm12nm(
                np.min(self.options['filter_range']))
            delta_nm =  orb.utils.spectrum.fwhm_cm12nm(
                self.get_lines_fwhm(),
                (np.min(self.options['filter_range'])
                 + np.max(self.options['filter_range']))/2.)
        else:
            nm_min = np.min(self.options['filter_range'])
            nm_max = np.max(self.options['filter_range'])
            delta_nm =  self.get_lines_fwhm()
        self.options['lines'] = self.optionfile.get_lines(
           nm_min, nm_max, delta_nm)
    
        self._print_msg('Searched lines (in nm): {}'.format(
            self.options['lines']))
        if self.options['wavenumber']:
            self.options['lines'] = orb.utils.spectrum.nm2cm1(
                self.options['lines'])
            self._print_msg('Searched lines (in {}): {}'.format(
                unit, self.options['lines']))
       
        # Lines shift
        self.options['object_velocity'] = 0.
        store_option_parameter('object_velocity', 'OBJECT_VELOCITY', float,
                               optional=True)
        self._print_msg('Mean object velocity: {} km.s-1'.format(
            self.options['object_velocity']))

        # cov lines
        store_option_parameter('cov_lines', 'COV_LINES', str, ',',
                               optional=True)

        if 'cov_lines' in self.options:
            self.cov_pos = self.options['cov_lines']
        else:
            self.cov_pos = True

        # HELIO
        if self.options['wcs']:
            helio_velocity = self.get_helio_velocity()
            self._print_msg(
                'Heliocentric velocity correction: {} km.s-1'.format(
                helio_velocity))
        else:
            helio_velocity = 0.
        
        # Total mean velocity
        self.options['mean_velocity'] = (
            self.options['object_velocity']
            - helio_velocity)
        
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

        ## Init spectral cube
        self.spectralcube = SpectralCube(
            self.options['spectrum_cube_path'],
            data_prefix=self._get_data_prefix(),
            project_header=self._get_project_fits_header(),
            wcs_header = self.wcs_header,
            overwrite=self.overwrite)

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
        return hdr
    
    def get_lines_fwhm(self):
        """Return the expected value of the lines FWHM in nm or in
        cm-1 depending on the cube axis unit.  
          
        .. seealso:: :py:meth:`orb.utils.spectrum.compute_line_fwhm`
        """
        step_nb = max(self.options['step_nb'] - self.zpd_index, self.zpd_index)
        
        return orb.utils.spectrum.compute_line_fwhm(
            step_nb, self.options['step'],
            self.options['order'], apod_coeff=self.options['apodization'],
            wavenumber=self.options['wavenumber'])

    def get_helio_velocity(self):
        """Return heliocentric velocity in km.s-1

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
                         obs_coords, silent=True).rvcorrect()[0]

    def get_lines_shift(self):
        """Return the expected line shift in nm or in cm-1 depending
        on the cube axis unit.

        .. seealso:: :py:meth:`orb.utils.spectrum.compute_line_shift`
        """
        if self.options['wavenumber']:
            axis = orb.utils.spectrum.create_cm1_axis(
                self.options['step_nb'], self.options['step'],
                self.options['order'])
        else:
            axis = orb.utils.spectrum.create_nm_axis(
                self.options['step_nb'], self.options['step'],
                self.options['order'])
            
        w_mean = (axis[-1] + axis[0]) / 2

        return orb.utils.spectrum.line_shift(
            self.options['mean_velocity'],
            w_mean,
            wavenumber=self.options['wavenumber'])
    
            
    def extract_lines_maps(self):
        """Extract lines maps.

        This is a wrapper around
        :py:meth:`orcs.SpectralCube.extract_lines_maps`.

        .. seealso:: :py:meth:`orcs.SpectralCube.extract_lines_maps`
        """
        if self.options.get('roi') is not None:
            x_range = self.options['roi'][:2]
            y_range = self.options['roi'][2:]
        else:
            x_range = None
            y_range = None

        if self.options['apodization'] == 1.0:
            fmodel = 'sinc'
        else:
            fmodel = 'gaussian'
            
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
            y_range=y_range,
            fmodel=fmodel,
            cov_pos=self.cov_pos)

    def get_sky_radial_velocity(self):
        """Return sky radial velocity

        This is a wrapper around
        :py:meth:`orcs.SpectralCube.get_sky_radial_velocity`.

        .. seealso:: :py:meth:`orcs.SpectralCube.get_sky_radial_velocity`
        """
        if self.options['apodization'] == 1.0:
            fmodel = 'sinc'
        else:
            fmodel = 'gaussian'
        
        self.spectralcube.get_sky_radial_velocity(
            self.options.get('sky_regions_path'),
            self.get_lines_fwhm(),
            self.options['filter_range'],
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            poly_order=self.options.get('poly_order'),
            calibration_laser_map_path=self.options[
                'calibration_laser_map_path'],
            nm_laser = self.config['CALIB_NM_LASER'],
            fmodel=fmodel)
        
    def extract_raw_lines_maps(self):
        """Extract raw lines maps.

        This is a wrapper around
        :py:meth:`orcs.SpectralCube.extract_raw_lines_maps`.

        .. seealso:: :py:meth:`orcs.SpectralCube.extract_raw_lines_maps`
        """
        self.spectralcube.extract_raw_lines_maps(
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            self.options['step'],
            self.options['order'],
            self.options['wavenumber'],
            calibration_laser_map_path=self.options[
                'calibration_laser_map_path'],
            nm_laser = self.config['CALIB_NM_LASER'])

    def extract_integrated_spectra(self):
        """Extract integrated spectra.

        This is a wrapper around
        :py:meth:`orcs.SpectralCube.extract_integrated_spectra`.

        .. seealso:: :py:meth:`orcs.SpectralCube.extract_integrated_spectra`
        """
        if self.options.get('roi') is not None:
            x_range = self.options['roi'][:2]
            y_range = self.options['roi'][2:]
        else:
            x_range = None
            y_range = None

        if self.options['apodization'] == 1.0:
            fmodel = 'sinc'
        else:
            fmodel = 'gaussian'
        
        self.spectralcube.extract_integrated_spectra(
            self.options['integ_reg_path'],
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
            fmodel=fmodel,
            cov_pos=self.cov_pos,
            plot=self.options['plot'],
            auto_sky_extraction=self.options['auto_sky_extraction'],
            sky_size_coeff=self.options['sky_size_coeff'])

    
        
#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCube(HDFCube):

    """ORCS spectral cube processing class.

    Fit and/or extract spectra.
    """

    def _get_maps_list_path(self):
        """Return path to the list of extacted emission lines maps"""
        return self._data_path_hdr + "maps_list"

    def _get_frame_header(self, file_type, comment=None):
        """Return frame header

        :param file_type: Type of file

        :param comment: (Optional) Comments on the file type (default
          None).
        """
        hdr = (self._get_basic_header('Deep frame')
               + self._project_header
               + self._get_basic_frame_header(self.dimx, self.dimy))
        hdr = self._add_wcs_header(hdr)
        return hdr
        
    def _add_wcs_header(self, hdr):
        """Add WCS header keywords to a header.

        :param hdr: Header to update
        """
        if self._wcs_header is not None:
            new_hdr = pyfits.Header()
            new_hdr.extend(hdr, strip=True,
                           update=True, end=True)

            new_hdr.extend(self._wcs_header, strip=True,
                           update=True, end=True)

            del new_hdr['RESTFRQ']
            del new_hdr['RESTWAV']
            del new_hdr['LONPOLE']
            del new_hdr['LATPOLE']

            return new_hdr
        else:
            return hdr

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


    def _get_deep_frame_path(self):
        return self._data_path_hdr + "deep_frame.fits"

    def _get_raw_line_map_path(self, line_nm):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "RAW_MAPS" + os.sep
                + basename + "%s_raw.fits"%str(line_nm))

    def _get_raw_err_map_path(self, line_nm):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "RAW_MAPS" + os.sep
                + basename + "%s_raw_err.fits"%str(line_nm))

    def _get_sky_mask_path(self):
        return self._data_path_hdr + "sky_mask.fits"

    def _get_noise_map_path(self):
        return self._data_path_hdr + "noise_map.fits"

    def _get_integrated_spectra_fit_params_path(self):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integ_spectra_fit_params")
    
    def _get_integrated_spectrum_path(self, region_name):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integrated_spectrum_{}.fits".format(region_name))

    def _get_integrated_spectrum_fit_path(self, region_name):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integrated_spectrum_fit_{}.fits".format(region_name))

    def _get_integrated_spectrum_header(self, region_name, axis, wavenumber):
        """Return integrated spectrum header

        :param axis: Axis of the spectrum
        
        :param region_name: Region name    
        """
        hdr = (self._get_basic_header('Integrated region {}'.format(region_name))
               + self._project_header
               + self._get_basic_spectrum_header(axis, wavenumber=wavenumber))
        return hdr

    def _get_fwhm_pix(self):
        """Return fwhm in channels from default lines FWHM"""
        return orb.utils.spectrum.nm2pix(
            self.nm_axis, self.nm_axis[0] + self.lines_fwhm)


    def _get_calibration_laser_map(self, calibration_laser_map_path):
        """Return the calibration laser map.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.
        """
        if calibration_laser_map_path is None:
            return np.ones((self.dimx, self.dimy), dtype=float)
        
        else:
            calibration_laser_map = self.read_fits(
                calibration_laser_map_path)
            if (calibration_laser_map.shape[0] != self.dimx):
                calibration_laser_map = orb.utils.image.interpolate_map(
                    calibration_laser_map, self.dimx, self.dimy)
            return calibration_laser_map

    def _get_calibration_coeff_map(self, calibration_laser_map_path, nm_laser):
        """Return the calibration coeff map based on the calibration
        laser map and the laser wavelength.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.

        :param nm_laser: calibration laser wavelength in nm.
        """
        if calibration_laser_map_path is None:
            return np.ones((self.dimx, self.dimy), dtype=float)
        else:
            return (self._get_calibration_laser_map(
                calibration_laser_map_path)
                    / nm_laser)
    
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
                        corr_axis = orb.utils.spectrum.create_cm1_axis(
                            data_col.shape[1], step, order, corr=corr)
                        data_col[icol, :] = orb.utils.vector.interpolate_axis(
                            data_col[icol, :], base_axis, 5,
                            old_axis=corr_axis)
                    else:
                        corr_axis = orb.utils.spectrum.create_nm_axis_ireg(
                            data_col.shape[1], step, order, corr=1./corr)
                        data_col[icol, :] = orb.utils.vector.interpolate_axis(
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

        # get range to check if a quadrants extraction is nescessary
        mask_x_proj = np.nanmax(mask, axis=1).astype(float)
        mask_x_proj[np.nonzero(mask_x_proj == 0)] = np.nan
        mask_x_proj *= np.arange(self.dimx)
        x_min = int(np.nanmin(mask_x_proj))
        x_max = int(np.nanmax(mask_x_proj))
        
        mask_y_proj = np.nanmax(mask, axis=0).astype(float)
        mask_y_proj[np.nonzero(mask_y_proj == 0)] = np.nan
        mask_y_proj *= np.arange(self.dimy)
        y_min = int(np.nanmin(mask_y_proj))
        y_max = int(np.nanmax(mask_y_proj))

        if (x_max - x_min < self.dimx / float(self.DIV_NB)
            and y_max - y_min < self.dimy / float(self.DIV_NB)):
            quadrant_extraction = False
            QUAD_NB = 1
            DIV_NB = 1
        else:
            quadrant_extraction = True
            QUAD_NB = self.QUAD_NB
            DIV_NB = self.DIV_NB

        if np.any(calibration_coeff_map != 1.):
            if wavenumber:
                base_axis = orb.utils.spectrum.create_cm1_axis(
                    self.dimz, step, order, corr=1.)
            else:
                base_axis  = orb.utils.spectrum.create_nm_axis_ireg(
                    self.dimz, step, order, corr=1.)

            for iquad in range(0, QUAD_NB):
                
                if quadrant_extraction:
                    # x_min, x_max, y_min, y_max are now used for quadrants boundaries
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
                            ii, int(self.dimx/float(DIV_NB))))
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
                           wavenumber, calibration_laser_map, nm_laser,
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
        def _fit_lines_in_column(data, calib_map_col, nm_laser, mask_col,
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
            

            for ij in range(data.shape[0]):
                if mask_col[ij]:

                    ## SUBSTRACT SPECTRUM #############
                    
                    if calib_map_col[ij] != nm_laser:
                        # convert lines wavelength/wavenumber into
                        # uncalibrated positions
                        if wavenumber:
                            cm1_axis = orb.utils.spectrum.create_cm1_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij]/nm_laser)
                            
                            # interpolate and substract
                            if substract is not None:
                                cm1_axis_base = orb.utils.spectrum.create_cm1_axis(data.shape[1], step, order, corr=1.)
                                data[ij,:] -= orb.utils.vector.interpolate_axis(
                                   substract, cm1_axis, 5,
                                   old_axis=cm1_axis_base)
                                
                        else:
                            nm_axis = orb.utils.spectrum.create_nm_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij]/nm_laser)
                           
                            # interpolate and substract
                            if substract is not None:
                                nm_axis_base = orb.utils.spectrum.create_nm_axis(data.shape[1], step, order, corr=1.)
                                data[ij,:] -= orb.utils.vector.interpolate_axis(
                                   substract, nm_axis, 5,
                                   old_axis=nm_axis_base)
                                
                    # basic substraction if no interpolation of the
                    # spectrum has to be done
                    elif substract is not None:
                        data[ij,:] - substract

                   
                    ## FIT #############################
                        
                    # get signal range
                    min_range = np.nanmin(filter_range)
                    max_range = np.nanmax(filter_range)
                    
                    if fmodel == 'sinc':
                        fit_function = orb.utils.spectrum.robust_fit_sinc_lines_in_spectrum
                    else:
                        fit_function = orb.utils.spectrum.fit_lines_in_spectrum
                        
                    try:
                        result_fit = fit_function(
                            data[ij,:],
                            lines,
                            step, order,
                            nm_laser,
                            calib_map_col[ij],
                            fwhm_guess=fwhm_guess,
                            cont_guess=None,
                            fix_cont=False,
                            poly_order=poly_order,
                            cov_pos=cov_pos,
                            cov_fwhm=cov_fwhm,
                            fix_fwhm=fix_fwhm, 
                            fmodel=fmodel,
                            signal_range=[min_range, max_range],
                            return_fitted_vector=False,
                            wavenumber=wavenumber)
                        
                    except Exception, e:
                        warnings.warn('Exception occured during fit: {}'.format(e))
                        result_fit = []
                        
                else: result_fit = []


                if result_fit != []: # reject fit with no associated error
                    if not 'lines-params-err' in result_fit:
                        result_fit = []
                    
                for iline in range(len(lines)):
                    if result_fit != []:
                        fit[ij,iline,:] = result_fit[
                            'lines-params'][iline, :]
                        err[ij,iline,:] = result_fit[
                            'lines-params-err'][iline, :]
                            
                        # return the wavelength/wavenumber instead of
                        # the position in the vector
                        if wavenumber:
                            err[ij,iline,2] = orb.utils.spectrum.pix2cm1(
                                cm1_axis, err[ij,iline,2] + fit[ij,iline,2])
                            fit[ij,iline,2] = orb.utils.spectrum.pix2cm1(
                                cm1_axis, fit[ij,iline,2])
                            err[ij,iline,2] -= fit[ij,iline,2]
                            
                        else:
                            err[ij,iline,2] = orb.utils.spectrum.pix2nm(
                                nm_axis, err[ij,iline,2] + fit[ij,iline,2])
                            fit[ij,iline,2] = orb.utils.spectrum.pix2nm(
                                nm_axis, fit[ij,iline,2])
                            err[ij,iline,2] -= fit[ij,iline,2]
                            
                        err[ij,iline,2] = abs(err[ij,iline,2])
                        
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


        # check substract spectrum
        if np.all(substract == 0.): substract = None
        
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

        # get min and max of mask (not useful now but preparing the
        # use of a ds9 made mask)
        masked_pixels = np.nonzero(mask)
        x_min_mask = np.nanmin(masked_pixels[0])
        x_max_mask = np.nanmax(masked_pixels[0])
        y_min_mask = np.nanmin(masked_pixels[1])
        y_max_mask = np.nanmax(masked_pixels[1])
            
        for iquad in range(0, self.QUAD_NB):
            # note that x_min, x_max ... are redefined as the quadrant
            # boundaries
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)

            # avoid loading quad with no pixel to fit in it
            if ((x_min_mask in range(x_min, x_max)
                 or x_max_mask in range(x_min, x_max))
                and
                (y_min_mask in range(y_min, y_max)
                 or y_max_mask in range(y_min, y_max))):

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
                              calibration_laser_map[x_min + ii + ijob,
                                                    y_min:y_max],
                              nm_laser,
                              mask[x_min + ii + ijob, y_min:y_max],
                              lines, fwhm_guess, filter_range,
                              poly_order, fix_fwhm, fmodel,
                              step, order, wavenumber,
                              cov_pos, cov_fwhm, substract), 
                        modules=("import numpy as np", 
                                 "import orb.utils.spectrum",
                                 "import orcs.utils as utils",
                                 "import warnings"),
                        depfuncs=(orb.utils.spectrum.fit_lines_in_spectrum,
                                  orb.utils.spectrum.robust_fit_sinc_lines_in_spectrum)))
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
                        ii, int(self.dimx/float(self.DIV_NB))))

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
    ##     sky_regions = orb.utils.misc.get_mask_from_ds9_region_file(
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
    ##                     fits_header=self._get_frame_header("Sky Mask"))
    ##     return mask_map



    def _convert_fit_result(self, fit_result, wavenumber, step, order,
                            axis):
        """Convert the result from
        :py:meth:`orb.utils.spectrum.fit_lines_in_vector` in
        wavelength/wavenumber.

        :param fit_result: Result from fit

        :param wavenumber: If True the spectrum is in wavenumber else it
          is in wavelength.

        :param step: Step size in nm

        :param order: Folding order

        :param axis: Axis of the spectrum in wavelength or wavenumber.
        """
        fit_params = np.copy(fit_result['lines-params'])
        err_params = np.copy(fit_result['lines-params-err'])
        # convert position and fwhm to wavelength/wavenumber
        if wavenumber:
            fit_params[:, 2] = orb.utils.spectrum.pix2cm1(
                axis, fit_params[:, 2])
            err_params[:, 2] = (
                orb.utils.spectrum.pix2cm1(axis, err_params[:, 2]
                                  + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
            fit_params[:, 3] = (
                orb.utils.spectrum.pix2cm1(axis, fit_params[:, 3]
                                  + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
            err_params[:, 3] = (
                orb.utils.spectrum.pix2cm1(axis, err_params[:, 3]
                                  + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
        else:
            fit_params[:, 2] = orb.utils.spectrum.pix2nm(
                axis, fit_params[:, 2])
            err_params[:, 2] = (
                orb.utils.spectrum.pix2nm(axis, err_params[:, 2]
                                 + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
            fit_params[:, 3] = (
                orb.utils.spectrum.pix2nm(axis, fit_params[:, 3]
                                 + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
            err_params[:, 3] = (
                orb.utils.spectrum.pix2nm(axis, err_params[:, 3]
                                 + fit_result['lines-params'][:, 2])
                - fit_params[:, 2])
            
        return fit_params, err_params

    def get_sky_radial_velocity(self, sky_regions_file_path,
                                lines_fwhm, filter_range,
                                wavenumber, step, order,
                                poly_order=0,
                                calibration_laser_map_path=None,
                                nm_laser=None,
                                fix_fwhm=False,
                                fmodel='gaussian', show=True):
        """
        Return the sky radial velocity.

        :sky_regions_file_path: Path to a ds9 region file giving the
          pixels where the sky spectrum has to be extracted.

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

        :param fix_fwhm: If True fix FWHM to its guess (default
          False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param show: (Optional) If True, show the sky spectrum and the
          fitted function (default True).
          
        .. note:: Use :py:class:`orb.core.Lines` to get the sky lines
          to fit.
        """
        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser)
                
        self._print_msg("Extracting sky median vector")
        median_sky_spectrum = self._extract_spectrum_from_region(
            orb.utils.misc.get_mask_from_ds9_region_file(
                sky_regions_file_path,
                x_range=[0, self.dimx],
                y_range=[0, self.dimy]),
            calibration_coeff_map,
            wavenumber, step, order)
        
        if wavenumber:
            cm1_axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order, corr=1.)
            filter_min_nm = orb.utils.spectrum.cm12nm(np.max(filter_range))
            filter_max_nm = orb.utils.spectrum.cm12nm(np.min(filter_range))
            lines_fwhm_nm = orb.utils.spectrum.fwhm_cm12nm(
                lines_fwhm, (cm1_axis[0] + cm1_axis[-1])/2.)
        else:
            nm_axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order, corr=1.)
            filter_min_nm = np.min(filter_range)
            filter_max_nm = np.max(filter_range)
            lines_fwhm_nm = lines_fwhm

        lines, lines_name = Lines().get_sky_lines(
            filter_min_nm, filter_max_nm, lines_fwhm_nm, get_names=True)
   
        if wavenumber:
            lines = orb.utils.spectrum.nm2cm1(lines)
            fwhm_guess_pix = orb.utils.spectrum.cm12pix(
                cm1_axis, cm1_axis[0] + lines_fwhm)
            filter_range_pix = orb.utils.spectrum.cm12pix(
                cm1_axis, filter_range)
            lines_pix = orb.utils.spectrum.cm12pix(cm1_axis, lines)
            axis = cm1_axis
        else:
            fwhm_guess_pix = orb.utils.spectrum.nm2pix(
                nm_axis, nm_axis[0] + lines_fwhm)
            filter_range_pix = orb.utils.spectrum.nm2pix(nm_axis, filter_range)
            lines_pix = orb.utils.spectrum.nm2pix(nm_axis, lines)
            axis = nm_axis

        raise Exception('Must be reimplemented')

        fit = orb.utils.spectrum.fit_lines_in_vector(
            median_sky_spectrum,
            lines_pix,
            fwhm_guess=fwhm_guess_pix,
            cont_guess=None,
            fix_cont=False,
            poly_order=poly_order,
            cov_pos=True,
            cov_fwhm=True,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            observation_params=[step, order],
            signal_range=filter_range_pix,
            return_fitted_vector=True,
            wavenumber=wavenumber)

        if fit != []:

            fit_params, err_params = self._convert_fit_result(
                fit, wavenumber, step, order, axis)

            # convert velocity to km.s-1
            velocities = orb.utils.spectrum.compute_radial_velocity(
                fit_params[:, 2], lines,
                wavenumber=wavenumber)
            
            if wavenumber:
                velocities_err = orb.utils.spectrum.compute_radial_velocity(
                    err_params[:, 2] + cm1_axis[0],
                    cm1_axis[0])
            else:
                velocities_err = orb.utils.spectrum.compute_radial_velocity(
                    err_params[:, 2] + nm_axis[0],
                    nm_axis[0])

        self._print_msg("Mean sky radial velocity : {} +/- {} km/s".format(
            np.mean(velocities), np.mean(velocities_err)))
    
        if show:
            import pylab as pl
            fig = pl.figure(figsize=(10, 6))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.72])
            
            if wavenumber:
                filter_range = filter_range[::-1]
                #axis = orb.utils.spectrum.cm12nm(axis)
                #lines = orb.utils.spectrum.cm12nm(lines)
                #filter_range = orb.utils.spectrum.cm12nm(filter_range)
            ylim = [np.nanmin(median_sky_spectrum),
                    np.nanmax(median_sky_spectrum) * 1.1]
            
            
            for iline in range(np.size(lines)):
               ax.axvline(x=lines[iline], alpha=0.5,
                          color='0.', linestyle=':')
                
                
            ax.plot(axis, median_sky_spectrum, color='0.5',
                    label = 'sky spectrum')
            ax.plot(axis, fit['fitted-vector'], color='0.',
                    linewidth=2., label='fit')

            # add lines name
            text_height = np.max(ylim) * 1.02
        
            
            for iline in range(np.size(lines)):
                s = lines_name[iline]
                if 'MEAN' in s: s = s[5:-1]
                
                print r'({})&{}&{:.2f}&{:.2f}\\'.format(
                    iline, s, orb.utils.spectrum.cm12nm(lines[iline]),
                    lines[iline])
                t = ax.text(lines[iline], text_height,
                            r'({}) $\lambda${:.2f}'.format(
                                iline, lines[iline]),
                            rotation='vertical',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            size='small')
                pl.draw()
            
            ax.set_xlim(filter_range)
            ax.set_ylim(ylim)
            
            ax.legend()
            pl.show()

        return median_sky_spectrum , fit_params, err_params

            
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
        lines += orb.utils.spectrum.line_shift(lines_velocity, lines,
                                      wavenumber=wavenumber)

        calibration_laser_map = self._get_calibration_laser_map(
            calibration_laser_map_path)

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser)
            
        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            self._print_msg("Extracting sky median vector")
            median_sky_spectrum = self._extract_spectrum_from_region(
                orb.utils.misc.get_mask_from_ds9_region_file(
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
            wavenumber, calibration_laser_map, nm_laser,
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
            axis = orb.utils.spectrum.create_cm1_axis(self.dimz, step, order)
        else:
            axis = orb.utils.spectrum.create_nm_axis(self.dimz, step, order)
    
        delta = abs(axis[1] - axis[0])

        for iline in range(np.size(lines)):
            if wavenumber:
                line_name = Lines().round_nm2ang(
                    orb.utils.spectrum.cm12nm(rest_frame_lines[iline]))
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
                fits_header=self._get_frame_header(
                    "Height Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'HEI'))

            # write height map err
            map_path = self._get_map_path(line_name,
                                          param='height',
                                          err=True)
            self.write_fits(
                map_path, results[iline][4],
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Height Map Error %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'HEI_ERR'))
            
            # write amp map
            map_path = self._get_map_path(line_name,
                                          param='amplitude')
            self.write_fits(
                map_path, results[iline][1],
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Amplitude Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'AMP'))

            # write amp map err
            map_path = self._get_map_path(line_name,
                                          param='amplitude',
                                          err=True)
            self.write_fits(
                map_path, results[iline][5],
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Amplitude Map Error%d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'AMP_ERR'))
            
            # write FWHM map
            map_path = self._get_map_path(line_name,
                                          param='fwhm')
            fwhm_map = results[iline][3] * delta
            self.write_fits(
                map_path, fwhm_map,
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
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
                fits_header=self._get_frame_header(
                    "FWHM Map Error %d"%line_name,
                    comment='in {}'.format(unit)))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'FWHM_ERR'))


            # write velocity map
                
            velocity = orb.utils.spectrum.compute_radial_velocity(
                results[iline][2], rest_frame_lines[iline],
                wavenumber=wavenumber)
  
            map_path = self._get_map_path(line_name,
                                          param='velocity')
            self.write_fits(
                map_path, velocity,
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Velocity Map %d"%line_name,
                    comment='in km/s'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'VEL'))

            # write velocity map err
            # compute radial velocity from shift

            velocity = np.abs(orb.utils.spectrum.compute_radial_velocity(
                rest_frame_lines[iline] + results[iline][6],
                rest_frame_lines[iline],
                wavenumber=wavenumber))
           
            map_path = self._get_map_path(line_name,
                                          param='velocity',
                                          err=True)
            self.write_fits(
                map_path, velocity,
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Velocity Map Error %d"%line_name,
                    comment='in km/s'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'VEL_ERR'))

            # write SNR map
            map_path = self._get_map_path(line_name,
                                          param='snr')
            self.write_fits(
                map_path, results[iline][9],
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "SNR Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'SNR'))

            # write chi-square map
            map_path = self._get_map_path(line_name,
                                          param='chi')
            self.write_fits(
                map_path, results[iline][8],
                overwrite=self.overwrite,
                fits_header=self._get_frame_header(
                    "Reduced chi-square Map %d"%line_name))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'CHI'))
                

    def extract_integrated_spectra(self, regions_file_path,
                                   lines, lines_velocity,
                                   lines_fwhm, filter_range,
                                   wavenumber, step, order,
                                   poly_order=0,
                                   calibration_laser_map_path=None,
                                   nm_laser=None,
                                   sky_regions_file_path=None,
                                   cov_pos=True, cov_fwhm=True,
                                   fix_fwhm=False,
                                   fmodel='gaussian', 
                                   plot=True,
                                   auto_sky_extraction=False,
                                   sky_size_coeff=2.):

        """
        Extract integrated spectra and their emission lines parameters.

        All parameters of each emission line are mapped. So that for a
        gaussian fit with 4 parameters (height, amplitude, shift,
        fwhm) 4 maps are created for each fitted emission line in the
        cube.

        :param regions_file_path: Path to a ds9 reg file giving the
          positions of the regions. Each region is considered as a
          different region.

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

        :param plot: (Optional) If True, plot each intergrated spectrum along
          with the its fit (default True).

        :param auto_sky_extraction: (Optional) If True and if no sky
          region file path is given the sky is taken around the region
          and substracted. The radius of the region around is
          controlled with sky_size_coeff. Note that this extraction
          only works for circular regions (default False).

        :param sky_size_coeff: (Optional) Coefficient ofthe
          external radius of the sky region (defautl 2.).
        """
        self._print_msg("Extracting integrated spectra", color=True)

        rest_frame_lines = np.copy(lines)
        lines += orb.utils.spectrum.line_shift(lines_velocity, lines,
                                      wavenumber=wavenumber)

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser)
        
        # Create parameters file
        paramsfile = orb.core.ParamsFile(
            self._get_integrated_spectra_fit_params_path())

        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            self._print_msg("Extracting sky median vector")
            median_sky_spectrum = self._extract_spectrum_from_region(
                orb.utils.misc.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy]),
                calibration_coeff_map,
                wavenumber, step, order)
        else:
            median_sky_spectrum = np.zeros(self.dimz, dtype=float)
            
        # extract regions
        integ_spectra = list()

        if wavenumber:
            cm1_axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order, corr=1.)
            fwhm_guess_pix = orb.utils.spectrum.cm12pix(
                cm1_axis, cm1_axis[0] + lines_fwhm)
            filter_range_pix = orb.utils.spectrum.cm12pix(cm1_axis, filter_range)
            axis = cm1_axis
            
        else:
            nm_axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order, corr=1.)
            fwhm_guess_pix = orb.utils.spectrum.nm2pix(
                nm_axis, nm_axis[0] + lines_fwhm)
            filter_range_pix = orb.utils.spectrum.nm2pix(nm_axis, filter_range)
            axis = nm_axis

        with open(regions_file_path, 'r') as f:
            region_index = -1
            for ireg in f:
                if len(ireg) > 3:
                    region = orb.utils.misc.get_mask_from_ds9_region_line(
                        ireg,
                        x_range=[0, self.dimx],
                        y_range=[0, self.dimy])
                    if len(region[0]) > 0:
                        region_index += 1
                        if auto_sky_extraction:
                            self._print_error('Not implemented yet')
                            # get sky around region
                            ## reg_bary = (np.mean(region[0]),
                            ##             np.mean(region[1]))
                            ## reg_size = max(np.max(region[0])
                            ##                - np.min(region[0]),
                            ##                np.max(region[1])
                            ##                - np.min(region[1]))

                            ## sky_size = reg_size * sky_size_coeff
                            ## x_min, x_max, y_min, y_max = (
                            ##     orb.utils.image.get_box_coords(
                            ##         reg_bary[0], reg_bary[1], sky_size,
                            ##         0, self.dimx, 0, self.dimy))

                            ## x_sky = list()
                            ## y_sky = list()
                            ## for ipix in range(int(x_min), int(x_max)):
                            ##     for jpix in range(int(y_min), int(y_max)):
                            ##         if ((math.sqrt((ipix - reg_bary[0])**2.
                            ##                        + (jpix - reg_bary[1])**2.)
                            ##              <= round(sky_size / 2.))):
                            ##             x_sky.append(ipix)
                            ##             y_sky.append(jpix)

                            ## sky = list([x_sky, y_sky])
                            ## mask = np.zeros((self.dimx, self.dimy))
                            ## mask[sky] = 1
                            ## mask[region] = 0
                            ## sky = np.nonzero(mask)

                        
                            ## median_sky_spectrum = (
                            ##     self._extract_spectrum_from_region(
                            ##         sky, median=True, silent=True))
                            
                        # extract spectrum
                        spectrum = self._extract_spectrum_from_region(
                            region, calibration_coeff_map,
                            wavenumber, step, order)
                        
                        if median_sky_spectrum is not None:
                             spectrum -= median_sky_spectrum

                        if wavenumber:
                            lines_pix = orb.utils.spectrum.cm12pix(
                                cm1_axis, lines)
                        else:
                            lines_pix = orb.utils.spectrum.nm2pix(
                                nm_axis, lines)

                        
                        # fit spectrum
                        fit = orb.utils.spectrum.fit_lines_in_vector(
                            spectrum,
                            lines_pix,
                            fwhm_guess=fwhm_guess_pix,
                            cont_guess=None,
                            fix_cont=False,
                            poly_order=poly_order,
                            cov_pos=cov_pos,
                            cov_fwhm=cov_fwhm,
                            fix_fwhm=fix_fwhm,
                            fmodel=fmodel,
                            observation_params=[step, order],
                            signal_range=filter_range_pix,
                            return_fitted_vector=True,
                            wavenumber=wavenumber)


                        if fit != []:
                            
                            # write spectrum and fit
                            if wavenumber:
                                spectrum_header = (
                                    self._get_integrated_spectrum_header(
                                    region_index, cm1_axis, wavenumber))
                            else:
                                spectrum_header = (
                                    self._get_integrated_spectrum_header(
                                    region_index, nm_axis, wavenumber))
                                
                            self.write_fits(
                                self._get_integrated_spectrum_path(
                                    region_index),
                                spectrum, fits_header=spectrum_header,
                                overwrite=self.overwrite)
                            self.write_fits(
                                self._get_integrated_spectrum_fit_path(
                                    region_index),
                                fit['fitted-vector'],
                                fits_header=spectrum_header,
                                overwrite=self.overwrite)

                            fit_params, err_params = self._convert_fit_result(
                                fit, wavenumber, step, order, axis)

                            # convert velocity to km.s-1
                            velocities = orb.utils.spectrum.compute_radial_velocity(
                                fit_params[:, 2], rest_frame_lines,
                                wavenumber=wavenumber)

                            if wavenumber:
                                velocities_err = orb.utils.spectrum.compute_radial_velocity(
                                    err_params[:, 2] + cm1_axis[0],
                                    cm1_axis[0])
                            else:
                                velocities_err = orb.utils.spectrum.compute_radial_velocity(
                                    err_params[:, 2] + nm_axis[0],
                                    nm_axis[0])

                            for iline in range(fit_params.shape[0]):
                                snr = abs(fit_params[iline, 1]
                                          / err_params[iline, 1])
                                if wavenumber:
                                    line_name = Lines().round_nm2ang(
                                        orb.utils.spectrum.cm12nm(
                                            rest_frame_lines[iline]))
                                else:
                                    line_name = Lines().round_nm2ang(
                                        rest_frame_lines[iline])
                                    

                                fit_results = {
                                    'reg_index': region_index,
                                    'line_name': line_name,
                                    'h': fit_params[iline, 0],
                                    'a': fit_params[iline, 1],
                                    'x': fit_params[iline, 2],
                                    'v': velocities[iline],
                                    'fwhm': fit_params[iline, 3],
                                    'h_err': err_params[iline, 0],
                                    'a_err': err_params[iline, 1],
                                    'x_err': err_params[iline, 2],
                                    'v_err': velocities[iline],
                                    'fwhm_err': err_params[iline, 3],
                                    'snr': snr}

                                paramsfile.append(fit_results)

                        if plot:
                            import pylab as pl
                            if wavenumber:
                                pl.plot(cm1_axis, spectrum,
                                        label='orig spectrum')
                                if fit != []:
                                    pl.plot(cm1_axis, fit['fitted-vector'],
                                            label='fit')
                            else:
                                pl.plot(nm_axis, spectrum,
                                        label='orig spectrum')
                                if fit != []:
                                    pl.plot(nm_axis, fit['fitted-vector'],
                                            label='fit')
                                    
                            pl.legend()
                            pl.show()
                            
                        integ_spectra.append(spectrum)
                    
        return integ_spectra
        

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
    ##         searched_lines = orb.utils.spectrum.nm2pix(
    ##             self.nm_axis, lines_nm),
    ##         signal_range=[orb.utils.spectrum.nm2pix(self.nm_axis,
    ##                                        self.filter_min),
    ##                       orb.utils.spectrum.nm2pix(self.nm_axis,
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
    ##                 gauss_line = orb.utils.spectrum.gaussian1d(
    ##                     np.arange(self.dimz),
    ##                     0., amp, dx, fwhm)
    ##                 fitted_spectrum += gauss_line
                    
    ##             fitted_cube[ii,ij,:] = fitted_spectrum
                    
    ##     return fitted_cube

    def extract_raw_lines_maps(self, lines, lines_velocity,
                               lines_fwhm, step, order, wavenumber,
                               calibration_laser_map_path=None,
                               nm_laser=None):
        """Raw extraction of a portion of the spectral cube. Return
        the mean of the slice.

        :param lines: Lines rest frame. Must be given in the unit of
          the cube (cm-1 if wavenumber is True, nm if False)

        :param lines_velocity: Mean Lines velocity in km.s-1.

        :param lines_fwhm: Expected FWHM of the lines. Must be given
          in the unit of the cube (cm-1 if wavenumber is True, nm if
          False)

        :param step: Step size in nm

        :param order: Folding order

        :param wavenumber: If True the cube is in wavenumber else it
          is in wavelength.
          
        :param calibration_laser_map_path: (Optional) If not None the
          cube is considered to be uncalibrated in wavelength. In this
          case the calibration laser wavelength (nm_laser) must also
          be given (default None).
          
        :param nm_laser: (Optional) Calibration laser wavelentgh. Must
          be given if calibration_laser_map_path is not None (default
          None).
        """

        def _extract_raw_lines_in_column(data, step, order,
                                         calib_map_col, lines, slice_size,
                                         wavenumber):

            col = np.empty((data.shape[0], len(lines)), dtype=float)
            # convert lines wavelength/wavenumber into calibrated
            # positions
            if np.all(calib_map_col == 1.):
                if wavenumber:
                    cm1_axis = orb.utils.spectrum.create_cm1_axis(
                        data.shape[1], step, order, corr=1.)
                    lines_pix = orb.utils.spectrum.cm12pix(cm1_axis, lines)
                else:
                    nm_axis = orb.utils.spectrum.create_nm_axis(
                        data.shape[1], step, order)
                    lines_pix = orb.utils.spectrum.nm2pix(nm_axis, lines)

            for ij in range(data.shape[0]):
                if calib_map_col[ij] != 1.:
                    # convert lines wavelength/wavenumber into
                    # uncalibrated positions
                    if wavenumber:
                        cm1_axis = orb.utils.spectrum.create_cm1_axis(
                            data.shape[1], step, order,
                            corr=calib_map_col[ij])
                        lines_pix = orb.utils.spectrum.cm12pix(cm1_axis, lines)
                    else:
                        nm_axis = orb.utils.spectrum.create_nm_axis(
                            data.shape[1], step, order,
                            corr=calib_map_col[ij])
                        lines_pix = orb.utils.spectrum.nm2pix(nm_axis, lines)
                        
                for iline in range(len(lines_pix)):
                    pix_min = int(lines_pix[iline] - slice_size/2.)
                    pix_max = pix_min + slice_size + 1.
                    col[ij, iline] = np.mean(data[ij, pix_min:pix_max])

            return col

        FWHM_COEFF = 2.5

        raw_maps = np.empty((self.dimx, self.dimy, len(lines)), dtype=float)
        raw_maps.fill(np.nan)

        rest_frame_lines = np.copy(lines)
        lines += orb.utils.spectrum.line_shift(lines_velocity, lines,
                                      wavenumber=wavenumber)

        if wavenumber:
            axis = orb.utils.spectrum.create_cm1_axis(self.dimz, step, order)
        else:
            axis = orb.utils.spectrum.create_nm_axis(self.dimz, step, order)

        axis_mean = (axis[0] + axis[-1]) / 2.
        
        if wavenumber:
            slice_min = orb.utils.spectrum.cm12pix(
                axis, axis_mean-FWHM_COEFF/2.*lines_fwhm)
            slice_max = orb.utils.spectrum.cm12pix(
                axis, axis_mean+FWHM_COEFF/2.*lines_fwhm) + 1
        else:
            slice_min = orb.utils.spectrum.nm2pix(
                axis, axis_mean - FWHM_COEFF/2. * lines_fwhm)
            slice_max = orb.utils.spectrum.nm2pix(
                axis, axis_mean + FWHM_COEFF/2. * lines_fwhm) + 1
            
        slice_size = math.floor(slice_max - slice_min)

        self._print_msg('Slice size: {} pixels'.format(slice_size))


        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser)

        for iquad in range(0, self.QUAD_NB):
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
                    _extract_raw_lines_in_column,
                    args=(iquad_data[ii+ijob,:,:],
                          step, order,
                          calibration_coeff_map[x_min + ii + ijob,
                                                y_min:y_max],
                          lines, slice_size, wavenumber), 
                    modules=("import numpy as np", 
                             "import orb.utils.spectrum",
                             "import orcs.utils as utils")))
                        for ijob in range(ncpus)]
            
                for ijob, job in jobs:
                    raw_maps[x_min+ii+ijob,y_min:y_max,:] = job()
                
                progress.update(ii, info="column : {}/{}".format(
                    ii, self.dimx))
                
            self._close_pp_server(job_server)
            progress.end()
        
        
        # write maps
        for iline in range(len(lines)):
            if wavenumber:
                line_name = Lines().round_nm2ang(
                    orb.utils.spectrum.cm12nm(rest_frame_lines[iline]))
                unit = 'cm-1'
            else:
                line_name = Lines().round_nm2ang(rest_frame_lines[iline])
                unit = 'nm'
                
            raw_path = self._get_raw_line_map_path(line_name)
            raw_header = self._get_frame_header(
                    'RAW LINE MAP {:d} {}'.format(line_name, unit))

            ## err_path = self._get_raw_err_map_path(line_name)
            ## err_header = self._get_frame_header(
            ##         'RAW ERR MAP {:d} {}'.format(line_name, unit))
            
            self.write_fits(raw_path, raw_maps[:,:,iline],
                            overwrite=self.overwrite,
                            fits_header=raw_header)
            ## self.write_fits(err_path, err_map, overwrite=self.overwrite,
            ##     fits_header=err_header)
        
        
        
        
