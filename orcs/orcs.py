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
import inspect
import scipy.interpolate
# import ORB
try:
    from orb.core import (
        Tools, OptionFile, Lines,
        ProgressBar, HDFCube)
    import orb.utils.spectrum
    import orb.utils.image
    import orb.utils.stats
    import orb.utils.filters
    import orb.utils.misc
    import orb.fit
    

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
    header = None
    
    def __init__(self, option_file_path,
                 spectrum_cube_path=None, **kwargs):
        """Init Orcs class.

        :param option_file_path: Path to the option file.

        :param spectral_cube_path: (Optional) Path to the spectral
          cube. If None, CUBEPATH keyword must be set in the option
          file.
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
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
            
            if value is None: # look in header for keyword not present
                              # in the option file
                if key in self.header:
                    value = cast(self.header[key])
            
                
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
                    "Keyword '{}' must be set in the option file".format(key))
                
                
        Tools.__init__(self, **kwargs)
        self.__version__ = version.__version__
        self.overwrite = True
        
        ## get config parameters
        store_config_parameter('OBS_LAT', float)
        store_config_parameter('OBS_LON', float)
        store_config_parameter('OBS_ALT', float)
        store_config_parameter("CALIB_NM_LASER", float)


        ## parse option file
        self.optionfile = OptionFile(option_file_path)
        
        ## get header/option file parameters
        if spectrum_cube_path is None:
            store_option_parameter('spectrum_cube_path', 'CUBEPATH', str)
            spectrum_cube_path = self.options['spectrum_cube_path']

        # load cube
        cube = HDFCube(spectrum_cube_path,
                       ncpus=self.ncpus,
                       config_file_name=self.config_file_name)
        self.header = cube.get_cube_header()
        
        # Observation parameters
        
        store_option_parameter('object_name', 'OBJECT', str)
        store_option_parameter('filter_name', 'FILTER', str)
        store_option_parameter('apodization', 'APODIZ', float)
       
        self.options['project_name'] = (
            self.options['object_name']
            + '_' + self.options['filter_name']
            + '_' + str(self.options['apodization'])
            + '.ORCS')
        
        store_option_parameter('step', 'STEP', float)
        store_option_parameter('order', 'ORDER', int)
        store_option_parameter('axis_corr', 'AXISCORR', float)

        # wavenumber
        store_option_parameter('wavetype', 'WAVTYPE', str)
        if self.options['wavetype'] == 'WAVELENGTH':
            self.options['wavenumber'] = False
            self._print_msg('Cube is in WAVELENGTH (nm)')
            unit = 'nm'
        else:
            self.options['wavenumber'] = True
            self._print_msg('Cube is in WAVENUMBER (cm-1)')
            unit = 'cm-1'
        
        # wavelength calibration
        store_option_parameter('wavelength_calibration', 'WAVCALIB', bool,
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
        
        ## Get WCS header
        cube = HDFCube(self.options['spectrum_cube_path'],
                       silent_init=True,
                       ncpus=self.ncpus,
                       config_file_name=self.config_file_name)
        
        self.wcs = pywcs.WCS(self.header, naxis=2)
        self.wcs_header = self.wcs.to_header()

        # get ZPD index
        self.options['step_nb'] = cube.dimz
        if 'ZPDINDEX' in cube.get_cube_header():
            self.zpd_index = cube.get_cube_header()['ZPDINDEX']
        else:
            self._print_error('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')

        
        store_option_parameter('target_ra', 'TARGETR', str, ':',
                               post_cast=float)
        store_option_parameter('target_dec', 'TARGETD', str, ':',
                               post_cast=float)
        store_option_parameter('target_x', 'TARGETX', float)
        store_option_parameter('target_y', 'TARGETY', float)
        store_option_parameter('obs_date', 'DATE-OBS', str, '-',
                               post_cast=int)
        store_option_parameter('hour_ut', 'HOUR_UT', str, ':',
                               post_cast=float, optional=True)
        if 'hour_ut' not in self.options:
            self.options['hour_ut'] = (0.,0.,0.)

        # optional parameters
        store_option_parameter('object_regions_path', 'OBJ_REG', str)
        store_option_parameter('sky_regions_path', 'SKY_REG', str,
                               optional=True)


        # Integrated spectra
        ## store_option_parameter('integ_reg_path', 'INTEG_REG_PATH', str,
        ##                        optional=True)
        
        self.options['auto_sky_extraction'] = False
        ## store_option_parameter('auto_sky_extraction', 'INTEG_AUTO_SKY', bool,
        ##                        optional=True)
        
        self.options['sky_size_coeff'] = 2.
        ## store_option_parameter('sky_size_coeff', 'INTEG_SKY_SIZE_COEFF', float,
        ##                        optional=True)

        self.options['plot'] = True
        ## store_option_parameter('plot', 'INTEG_PLOT', bool,
        ##                        optional=True)

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

        # signal range
        store_option_parameter('signal_range', 'NM_RANGE', str, ',',
                               optional=True, post_cast=float)

        if 'signal_range' in self.options:
            if self.options['wavenumber']:
                self.options['signal_range'] = orb.utils.spectrum.nm2cm1(
                    self.options['signal_range'])
        
            self._print_msg('Signal range: {}-{}'.format(
                np.min(self.options['signal_range']),
                np.max(self.options['signal_range'])))
        


        # HELIO
        helio_velocity = self.get_helio_velocity()
        self._print_msg(
            'Heliocentric velocity correction: {} km.s-1'.format(
                helio_velocity))
        
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
            overwrite=self.overwrite,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus)
        
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
            self.options['order'], apod_coeff=float(self.options['apodization']),
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
        if self.options['apodization'] == 1.0:
            fmodel = 'sinc'
        else:
            fmodel = 'gaussian'
        
        if 'signal_range' in self.options:
            signal_range = self.options['signal_range']
        else:
            signal_range = self.options['filter_range']

        self.spectralcube.extract_lines_maps(
            self.options['object_regions_path'],
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            signal_range,
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            axis_corr=self.options['axis_corr'],
            poly_order=self.options.get('poly_order'),
            calibration_laser_map_path=self.options[
                'calibration_laser_map_path'],
            nm_laser = self.config['CALIB_NM_LASER'],
            sky_regions_file_path=self.options.get('sky_regions_path'),
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
            sky_size_coeff=self.options['sky_size_coeff'],
            axis_corr=self.options['axis_corr'])

    
        
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
        hdr = (self._get_basic_header(file_type)
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



    def _get_calibration_laser_map(self, calibration_laser_map_path,
                                   nm_laser, axis_corr=1):
        """Return the calibration laser map.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.
    
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """
        if calibration_laser_map_path is None:
            return (np.ones((self.dimx, self.dimy), dtype=float)
                    * nm_laser * axis_corr)
        
        else:
            calibration_laser_map = self.read_fits(
                calibration_laser_map_path)
            if (calibration_laser_map.shape[0] != self.dimx):
                calibration_laser_map = orb.utils.image.interpolate_map(
                    calibration_laser_map, self.dimx, self.dimy)
            return calibration_laser_map

    def _get_calibration_coeff_map(self, calibration_laser_map_path, nm_laser,
                                   axis_corr=1):
        """Return the calibration coeff map based on the calibration
        laser map and the laser wavelength.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.

        :param nm_laser: calibration laser wavelength in nm.

        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.

        """
        return (self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)
                / nm_laser)
    
    def _extract_spectrum_from_region(self, region,
                                      calibration_coeff_map,
                                      wavenumber, step, order,
                                      subtract_spectrum=None,
                                      median=False, silent=False):
        """
        Extract the mean spectrum from a region of the cube.
        
        :param region: A list of the indices of the pixels integrated
          in the returned spectrum.

        :param calibration_coeff_map: Map of the calibration
          coefficient (calibration laser map / calibration laser
          wavelength). If the cube is wavelength calibrated this map
          must be a map of ones.

        :param wavenumber: If True cube is in wavenumber, else it is
          in wavelength.

        :param step: Step size in nm

        :param order: Folding order  
    
        :subtract_spectrum: Remove the given spectrum from the
          extracted spectrum before fitting parameters. Useful
          to remove sky spectrum. Both spectra must have the same size.

        :param median: If True the integrated spectrum is the median
          of the spectra. Else the integrated spectrum is the mean of
          the spectra (Default False).

        :param silent: (Optional) If True, nothing is printed (default
          False).
          
        :return: A scipy.UnivariateSpline object.
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
                        corr_axis = orb.utils.spectrum.create_nm_axis(
                            data_col.shape[1], step, order, corr=corr)
                        data_col[icol, :] = orb.utils.vector.interpolate_axis(
                            data_col[icol, :], base_axis, 5,
                            old_axis=corr_axis)
                else:
                    data_col[icol, :].fill(np.nan)
                    
            if median:
                return bn.nanmedian(data_col, axis=0), 1
            else:
                return bn.nansum(data_col, axis=0), np.nansum(mask_col)
            
            

        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1

        spectrum = np.zeros(self.dimz, dtype=float)
        counts = 0
        
        # get range to check if a quadrants extraction is necessary
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

        calibration_coeff_center = calibration_coeff_map[
            calibration_coeff_map.shape[0]/2,
            calibration_coeff_map.shape[1]/2]
        
        if wavenumber:
            base_axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order, corr=calibration_coeff_center)
        else:
            base_axis  = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order, corr=calibration_coeff_center)

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
                             'import orb.utils.spectrum',
                             'import orb.utils.vector'),
                    depfuncs=()))
                        for ijob in range(ncpus)]
                for ijob, job in jobs:
                    spec_to_add, spec_nb = job()
                    if not np.all(np.isnan(spec_to_add)):
                        spectrum += spec_to_add
                        counts += spec_nb

                if not silent:
                    progress.update(ii, info="column : {}/{}".format(
                        ii, int(self.dimx/float(DIV_NB))))
            self._close_pp_server(job_server)
            if not silent: progress.end()
            spectrum /= counts


        if subtract_spectrum is not None:
            spectrum -= subtract_spectrum

        spectrum_function = scipy.interpolate.UnivariateSpline(
            base_axis, spectrum, s=0, k=1, ext=1)
         
        return spectrum_function
        
    def _fit_lines_in_cube(self, region, lines, step, order,
                           fwhm_guess, wavenumber,
                           calibration_laser_map, nm_laser,
                           poly_order, filter_range, shift_guess,
                           fix_fwhm=False, fmodel='gaussian',
                           subtract=None, cov_pos=True,
                           cov_fwhm=True):
        
        """Fit lines in a spectral cube.

        :param region: A list of the indices of the pixels to extract.

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

        :param shift_guess: Velocity shift guess in km/s

        :param fix_fwhm: (Optional) If True FWHM is fixed to
          fwhm_guess value (defautl False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param subtract: (Optional) If not None this spectrum will be
          subtracted before fitting (must be a scipy.UnivariateSpline
          object).

        :param cov_pos: (Optional) Lines positions in each spectrum
          are shifted by the same value.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.
        """
        def _fit_lines_in_column(data, calib_map_col, nm_laser, mask_col,
                                 lines, fwhm_guess, filter_range,
                                 poly_order, fix_fwhm, fmodel,
                                 step, order, wavenumber,
                                 cov_pos, cov_fwhm, subtract,
                                 shift_guess):
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

                    ## SUBTRACT SPECTRUM #############
                    if subtract is not None:
                        # convert lines wavelength/wavenumber into
                        # uncalibrated positions
                        if wavenumber:
                            axis = orb.utils.spectrum.create_cm1_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij]/nm_laser).astype(float)
                        else:
                            axis = orb.utils.spectrum.create_nm_axis(
                                data.shape[1], step, order,
                                corr=calib_map_col[ij]/nm_laser).astype(float)
                            
                        # interpolate and subtract                       
                        data[ij,:] -= subtract(axis)
                
                    ## FIT #############################
                        
                    # get signal range
                    min_range = np.nanmin(filter_range)
                    max_range = np.nanmax(filter_range)
                    
                    # adjust fwhm with incident angle
                    ifwhm_guess = fwhm_guess * calib_map_col[ij] / nm_laser
                        
                    try:
                        result_fit = orb.fit.fit_lines_in_spectrum(
                            data[ij,:],
                            lines,
                            step, order,
                            nm_laser,
                            calib_map_col[ij],
                            fwhm_guess=ifwhm_guess,
                            shift_guess=shift_guess,
                            cont_guess=None,
                            poly_order=poly_order,
                            cov_pos=cov_pos,
                            cov_fwhm=cov_fwhm,
                            fix_fwhm=fix_fwhm, 
                            fmodel=fmodel,
                            signal_range=[min_range, max_range],
                            wavenumber=wavenumber)
                        
                    except Exception, e:
                        warnings.warn('Exception occured during fit: {}'.format(e))
                        import traceback
                        print traceback.format_exc()
                        
                        result_fit = []
                        
                else: result_fit = []


                if result_fit != []: # reject fit with no associated error
                    if not 'lines-params-err' in result_fit:
                        result_fit = []
                
                for iline in range(len(lines)):
                    if result_fit != []:
                        
                        # print result_fit['velocity']
                        # import pylab as pl
                        # pl.plot(data[ij,:])
                        # pl.plot(result_fit['fitted-vector'])
                        # pl.show()
                        # quit()
                        
                        fit[ij,iline,:] = result_fit[
                            'lines-params'][iline, :]
                        err[ij,iline,:] = result_fit[
                            'lines-params-err'][iline, :]
                        fit[ij,iline,2] = result_fit[
                            'velocity'][iline]
                        err[ij,iline,2] = result_fit[
                            'velocity-err'][iline]
                        
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

        ## init LineMaps object
        linemaps = LineMaps(self.dimx, self.dimy, lines, wavenumber,
                            project_header=self._project_header,
                            wcs_header=self._wcs_header,
                            config_file_name=self.config_file_name,
                            data_prefix=self._data_prefix,
                            ncpus=self.ncpus) 
        
        # check subtract spectrum
        if np.all(subtract == 0.): subtract = None
                    
        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[region] = True

        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)

            # avoid loading quad with no pixel to fit in it
            if np.any(mask[x_min:x_max, y_min:y_max]):
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
                              cov_pos, cov_fwhm, subtract, shift_guess), 
                        modules=("import numpy as np",
                                 "import orb.fit",
                                 "import orb.utils.spectrum",
                                 "import orcs.utils as utils",
                                 "import warnings", 'import math',
                                 'import inspect')))
                            for ijob in range(ncpus)]

                    for ijob, job in jobs:
                        (fit, err, chi) = job()
                        x_range = (x_min+ii+ijob, x_min+ii+ijob + 1)
                        y_range = (y_min, y_max)
                        linemaps.set_map('height', fit[:,:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude', fit[:,:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity', fit[:,:,2],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm', fit[:,:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('height-err', err[:,:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude-err', err[:,:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity-err', err[:,:,2],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm-err', err[:,:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('chi-square', chi[:,:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('snr', chi[:,:,1],
                                         x_range=x_range, y_range=y_range)

                    progress.update(ii, info="column : {}/{}".format(
                        ii, int(self.dimx/float(self.DIV_NB))))

                self._close_pp_server(job_server)
                progress.end()
    
        return linemaps


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


    def get_sky_radial_velocity(self, sky_regions_file_path,
                                lines_fwhm, filter_range,
                                wavenumber, step, order,
                                poly_order=0,
                                calibration_laser_map_path=None,
                                nm_laser=None,
                                fix_fwhm=False,
                                fmodel='gaussian', show=True,
                                axis_corr=1.):
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
    
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
          
        .. note:: Use :py:class:`orb.core.Lines` to get the sky lines
          to fit.
        """
        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)
                
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
            poly_order=poly_order,
            cov_pos=True,
            cov_fwhm=True,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            observation_params=[step, order],
            signal_range=filter_range_pix,
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
            
    def extract_lines_maps(self, object_regions_file_path, lines,
                           lines_velocity,
                           lines_fwhm, filter_range,
                           wavenumber, step, order,
                           poly_order=0,
                           calibration_laser_map_path=None,
                           nm_laser=None,
                           sky_regions_file_path=None,
                           cov_pos=True, cov_fwhm=True,
                           fix_fwhm=False,
                           fmodel='gaussian',
                           axis_corr=1.):
        
        """
        Extract emission lines parameters maps from a fit.

        All parameters of each emission line are mapped. So that for a
        gaussian fit with 4 parameters (height, amplitude, shift,
        fwhm) 4 maps are created for each fitted emission line in the
        cube.

        :param object_regions_file_path: Path to a ds9 regions file
          decribing the regions to extract.

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
          extracted. This spectrum will be subtracted to the spectral
          cube before it is fitted (default None).

        :param fix_fwhm: If True fix FWHM to its guess (default
          False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param cov_pos: (Optional) Lines positions in each spectrum
          are shifted by the same value.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.

        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """
        self._print_msg("Extracting lines data", color=True)
        
        calibration_laser_map = self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)
            
        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            self._print_msg("Extracting sky median vector")
            sky_region = orb.utils.misc.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy])
            median_sky_spectrum = self._extract_spectrum_from_region(
                sky_region,
                calibration_coeff_map,
                wavenumber, step, order)

            ## import pylab as pl
            ## base_axis = orb.utils.spectrum.create_nm_axis(
            ##     self.dimz, step, order, corr=axis_corr)
            ## pl.plot(base_axis, median_sky_spectrum(base_axis.astype(float)))
            ## pl.show()
        else:
            median_sky_spectrum = np.zeros(self.dimz, dtype=float)

    
        ## FIT ###
        self._print_msg("Fitting data")
        region = orb.utils.misc.get_mask_from_ds9_region_file(
                    object_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy])
        
        linemaps = self._fit_lines_in_cube(
            region, lines, step, order, lines_fwhm,
            wavenumber, calibration_laser_map, nm_laser,
            poly_order, filter_range, lines_velocity,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            cov_pos=cov_pos, cov_fwhm=cov_fwhm,
            subtract=median_sky_spectrum)

        ## SAVE MAPS ###
        linemaps.write_maps()            
                

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
                                   sky_size_coeff=2., axis_corr=1.):

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
          extracted. This spectrum will be subtracted to the spectral
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
          and subtracted. The radius of the region around is
          controlled with sky_size_coeff. Note that this extraction
          only works for circular regions (default False).

        :param sky_size_coeff: (Optional) Coefficient ofthe
          external radius of the sky region (default 2.).
          
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """
        self._print_msg("Extracting integrated spectra", color=True)

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)

        
        calibration_laser_map = self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)
                
        # Create parameters file
        paramsfile = orb.core.ParamsFile(
            self._get_integrated_spectra_fit_params_path())
        print 
        
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
            median_sky_spectrum = None
            
        # extract regions
        integ_spectra = list()

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
                        icorr = np.nanmean(calibration_coeff_map[region])
                        icalib = np.nanmean(calibration_laser_map[region])
                        
                        if wavenumber:
                            axis = orb.utils.spectrum.create_cm1_axis(
                                self.dimz, step, order, corr=icorr)
                        else:
                            axis = orb.utils.spectrum.create_nm_axis(
                                self.dimz, step, order, corr=icorr)

                        if auto_sky_extraction:
                            # subtract sky around region
                            self._print_error('Not implemented yet')              
                            
                        # extract spectrum
                        spectrum = self._extract_spectrum_from_region(
                            region, calibration_coeff_map,
                            wavenumber, step, order)
                        # interpolate
                        spectrum = spectrum(axis.astype(float))
                        
                        if median_sky_spectrum is not None:
                             spectrum -= median_sky_spectrum(axis)

                        # get signal range
                        min_range = np.nanmin(filter_range)
                        max_range = np.nanmax(filter_range)
                    
                        # adjust fwhm with incident angle
                        ilines_fwhm = lines_fwhm * icorr
                   
                        # fit spectrum                            
                        try:
                            result_fit = orb.fit.fit_lines_in_spectrum(
                                spectrum,
                                lines,
                                step, order,
                                nm_laser,
                                icalib,
                                fwhm_guess=ilines_fwhm,
                                shift_guess=lines_velocity,
                                cont_guess=None,
                                poly_order=poly_order,
                                cov_pos=cov_pos,
                                cov_fwhm=cov_fwhm,
                                fix_fwhm=fix_fwhm,
                                fmodel=fmodel,
                                signal_range=[min_range, max_range],
                                wavenumber=wavenumber)
                            
                        except Exception, e:
                            warnings.warn('Exception occured during fit: {}'.format(e))
                            import traceback
                            print traceback.format_exc()
                            
                            result_fit = []
                        

                        if result_fit != []:
                            
                            # write spectrum and fit
                            spectrum_header = (
                                self._get_integrated_spectrum_header(
                                    region_index, axis, wavenumber))
                                
                            self.write_fits(
                                self._get_integrated_spectrum_path(
                                    region_index),
                                spectrum, fits_header=spectrum_header,
                                overwrite=self.overwrite)
                            self.write_fits(
                                self._get_integrated_spectrum_fit_path(
                                    region_index),
                                result_fit['fitted-vector'],
                                fits_header=spectrum_header,
                                overwrite=self.overwrite)

                            fit_params = result_fit['lines-params']
                            err_params = result_fit['lines-params-err']
                            velocities = result_fit['velocity']
                            velocities_err = result_fit['velocity-err'] 
                            snr = result_fit['snr'] 
                            
                            
                            for iline in range(fit_params.shape[0]):
                                if wavenumber:
                                    line_name = Lines().round_nm2ang(
                                        orb.utils.spectrum.cm12nm(
                                            lines[iline]))
                                else:
                                    line_name = Lines().round_nm2ang(
                                        lines[iline])
                                    

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
                                    'v_err': velocities_err[iline],
                                    'fwhm_err': err_params[iline, 3],
                                    'snr': snr[iline]}

                                paramsfile.append(fit_results)

                            if plot:
                                import pylab as pl
                                pl.plot(axis, spectrum,
                                        label='orig spectrum')
                                pl.plot(axis, result_fit['fitted-vector'],
                                        label='fit')

                                pl.grid()
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
    ##         detect_coeff=0.)

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
                               nm_laser=None, axis_corr=1.):
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
    
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
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
            calibration_laser_map_path, nm_laser, axis_corr=axis_corr)

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
        
        


##################################################
#### CLASS LineMaps ##############################
##################################################

class LineMaps(Tools):
    """Manage line parameters maps"""


    params = ('height', 'height-err', 'amplitude', 'amplitude-err',
              'velocity', 'velocity-err', 'fwhm', 'fwhm-err',
              'chi-square', 'snr')

    _wcs_header = None

    def __init__(self, dimx, dimy, lines, wavenumber,
                 project_header=None, wcs_header=None,
                 **kwargs):
        """Init class
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
        """
        Tools.__init__(self, **kwargs)
        self._project_header = project_header
        self._wcs_header = wcs_header
        self.__version__ = version.__version__
        self.dimx = dimx
        self.dimy = dimy
        self.wavenumber = wavenumber
        
        # Create dataset
        if np.size(lines) == 1:
            self.lines = np.array([np.squeeze(lines)])
        else:
            self.lines = lines

        if self.wavenumber:
            self.line_names = Lines().round_nm2ang(
                orb.utils.spectrum.cm12nm(self.lines))
            self.unit = 'cm-1'
        else:
            self.line_names = Lines().round_nm2ang(
                self.lines)
            self.unit = 'nm'

        if np.size(self.line_names) == 1:
            self.line_names = np.array([np.squeeze(self.line_names)])

        self.data = dict()
        base_array =  np.empty((self.dimx, self.dimy, len(lines)),
                               dtype=float)
        base_array.fill(np.nan)
        for iparam in self.params:
            self.data[iparam] = np.copy(base_array)


    def _get_map_path(self, line_name, param):
        """Return the path to a map of one gaussian fit parameter for
        one given emission line.

        :param line_name: Name of the emission line

        :param param: (Optional) Parameter
          
        """
        if param not in self.params:
            self._print_error('Bad parameter')
         
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "MAPS" + os.sep
                + basename + "map.{}.{}.fits".format(line_name, param))


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

        
    def set_map(self, param, data_map, x_range=None, y_range=None):

        if param not in self.params:
            self._print_error('Bad parameter')
            
        if x_range is None and y_range is None:
            self.data[param] = data_map
        else:
            self.data[param][
                min(x_range):max(x_range),
                min(y_range):max(y_range)] = data_map

    def write_maps(self):

        for param in self.params:
            
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
