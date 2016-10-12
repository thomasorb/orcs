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

# import core
from core import HDFCube, Tools, OrcsBase
from process import SpectralCubeFitter

# import ORB
import orb.core
import orb.utils.spectrum
import orb.utils.stats
import orb.utils.filters
import orb.utils.misc
    

from rvcorrect import RVCorrect

##################################################
#### CLASS Orcs ##################################
##################################################

class Orcs(OrcsBase):
    """ORCS user-interface.

    Manage the option file and the file system. This is the equivalent
    in ORCS of :py:meth:`orbs.Orbs` for ORBS::

      
      CUBEPATH /path/to/CALIBRATED_SPECTRUM # Path to the spectral
                                            # cube (hdf5 format)
      
      LINES [NII]6548,Halpha,[NII]6583,[SII]6716,[SII]6731 # Lines to fit
      
      OBJECT_VELOCITY 170 # in km.s-1, mean velocity of the object
      
      POLY_ORDER 0 # Order of the polynomial used to fit continuum

   
    Option file keywords:
    
    :CUBEPATH: Path to the spectrum cube (hdf5 format)
    
    :LINES: Wavelength (in nm) of the lines to fit separated by a
      comma. Their full name can also be given but it must respect the
      list of recorded lines by :py:class:`orb.core.Lines`. A line
      wavelength in nm can also be given.
      
    :OBJECT_VELOCITY: (Optional) Mean velocity of the observed object
      in km.s-1. This is used to guess the mean shift of the lines.

    :POLY_ORDER: (Optional) Order of the polynomial used to fit
      continuum. Be extremely careful with high order polynomials.

    :SKY_REG: (Optional) Path to a ds9 region file defining the sky pixels.

    .. seealso:: :py:class:`orb.core.OptionFile`

    """
    
    def __init__(self, option_file_path,
                 spectrum_cube_path=None, **kwargs):
        """Init Orcs class.

        :param option_file_path: Path to the option file.

        :param spectral_cube_path: (Optional) Path to the spectral
          cube. If None, CUBEPATH keyword must be set in the option
          file.
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
        """
        
        ## parse option file
        self.optionfile = orb.core.OptionFile(option_file_path)

        ## get header / optionfile parameters
        if spectrum_cube_path is None:
            self._store_option_parameter('spectrum_cube_path', 'CUBEPATH', str)
            spectrum_cube_path = self.options['spectrum_cube_path']

        ## Base init
        OrcsBase.__init__(self, spectrum_cube_path, **kwargs)

    
        # optional parameters
        self._store_option_parameter('object_regions_path', 'OBJ_REG', str)
        self._store_option_parameter('sky_regions_path', 'SKY_REG', str,
                               optional=True)


        # Integrated spectra
        self._store_option_parameter('integ_reg_path', 'INTEG_REG_PATH', str,
                               optional=True)
        
        self.options['auto_sky_extraction'] = False
        ## self._store_option_parameter('auto_sky_extraction', 'INTEG_AUTO_SKY', bool,
        ##                        optional=True)
        
        self.options['sky_size_coeff'] = 2.
        ## self._store_option_parameter('sky_size_coeff', 'INTEG_SKY_SIZE_COEFF', float,
        ##                        optional=True)

        self.options['plot'] = True
        ## self._store_option_parameter('plot', 'INTEG_PLOT', bool,
        ##                        optional=True)

        # filter boundaries
        self._store_option_parameter('filter_range', 'RANGE_NM', str, ',',
                               post_cast=float, optional=True)
        if 'filter_range' not in self.options:
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
                self.unit, self.options['lines']))
       
        # Lines shift
        self.options['object_velocity'] = 0.
        self._store_option_parameter('object_velocity', 'OBJECT_VELOCITY', str, ',',
                               optional=True, post_cast=float)
        self.options['object_velocity'] = np.array(
            self.options['object_velocity'], dtype=float)
        
        self._print_msg('Mean object velocity: {} km.s-1'.format(
            self.options['object_velocity']))

        # velocity range
        self.options['velocity_range'] = None
        self._store_option_parameter('velocity_range', 'VELOCITY_RANGE', float,
                               optional=True)
        self._print_msg('Velocity range: {} km/s'.format(
            self.options['velocity_range']))


        # cov lines
        self._store_option_parameter('cov_lines', 'COV_LINES', str, ',',
                               optional=True)
        if 'cov_lines' in self.options:
            self.cov_pos = self.options['cov_lines']
            # put all sky lines with the same cov symbol
            _lines = self.optionfile.get('LINES', str)
            _lines = _lines.strip().split(',')
            if len(_lines) != len(self.cov_pos):
                self._print_error('The number of covariance symbols must equal the number of lines (only 1 number for SKY)')
            if 'SKY' in _lines:
                # get symbol associated with SKY keyword
                sym_sky_index = _lines.index('SKY')
                sym_sky = self.cov_pos[sym_sky_index]
                cov_pos = list(self.cov_pos)
                cov_pos.pop(sym_sky_index)
                cov_pos = cov_pos[:sym_sky_index] + list([sym_sky]) * (len(self.options['lines']) - len(_lines) + 1) + cov_pos[sym_sky_index:]
                self.cov_pos = cov_pos
                
               
        else:
            self.cov_pos = True

        
        # cov sigma
        self._store_option_parameter('cov_sigma', 'COV_SIGMA', str, ',',
                               optional=True)

        if 'cov_sigma' in self.options:
            self.cov_sigma = self.options['cov_sigma']
            if 'SKY' in _lines:
                # get symbol associated with SKY keyword
                sym_sky_index = _lines.index('SKY')
                sym_sky = self.cov_sigma[sym_sky_index]
                cov_sigma = list(self.cov_sigma)
                cov_sigma.pop(sym_sky_index)
                cov_sigma = cov_sigma[:sym_sky_index] + list([sym_sky]) * (len(self.options['lines']) - len(_lines) + 1) + cov_sigma[sym_sky_index:]
                self.cov_sigma = cov_sigma
        else:
            self.cov_sigma = True

        # signal range
        self._store_option_parameter('signal_range', 'NM_RANGE', str, ',',
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
            self.options['mean_velocity'], self.get_lines_shift(), self.unit))

        
        self._print_msg('Expected lines FWHM: {:.3f} {}'.format(
            self.get_lines_fwhm(), self.unit))
        
        # Continuum polynomial
        self.options['poly_order'] = 0
        self._store_option_parameter('poly_order', 'POLY_ORDER', int,
                               optional=True)
        self._print_msg(
            'Order of the polynomial used to fit continuum: {}'.format(
                self.options['poly_order']))

        # Force lines fitting model
        self._store_option_parameter('fmodel', 'FMODEL', str,
                               optional=True)
        if 'fmodel' in self.options:
            self.options['fmodel'] =  self.options['fmodel'].lower()
            self._print_msg(
                'Lines model: {}'.format(
                    self.options['fmodel']))

        # Binning
        self._store_option_parameter('binning', 'BINNING', int,
                               optional=True)
        self._print_msg('Binning: {}'.format(self.options['binning']))


        ## Init spectral cube
        self.spectralcube = SpectralCubeFitter(
            self.options['spectrum_cube_path'],
            data_prefix=self._get_data_prefix(),
            project_header=self._get_project_fits_header(),
            wcs_header = self.wcs_header,
            overwrite=self.overwrite,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus)

    def _store_option_parameter(self, option_key, key, cast, split=None,
                               optional=False, folder=False,
                               post_cast=str):
        """Store an optional parameter

        :param option_key: Option Key

        :param key: Option file key

        :param cast: Type cast

        :param split: (Optional) split key value after first cast (default None)

        :param optional: (Optional) Key is optional (no error raised
          if key does not exist) (default None)

        :param folder: (Optional) key type is a folder path (default
          False)

        :param post_cast: (Optional) Final cast on the key after first
          cast and split. Used only if split is nt None (default str).
        """

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

    
    def get_lines_fwhm(self):
        """Return the expected value of the lines FWHM in nm or in
        cm-1 depending on the cube axis unit.  
          
        .. seealso:: :py:meth:`orb.utils.spectrum.compute_line_fwhm`
        """
        step_nb = max(self.options['step_nb'] - self.zpd_index, self.zpd_index)

        # with sincgauss model FWHM must be set to the real one
        return orb.utils.spectrum.compute_line_fwhm(
            step_nb, self.options['step'],
            self.options['order'],
            #apod_coeff=float(self.options['apodization'],
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
            self.options['mean_velocity'][0],
            w_mean,
            wavenumber=self.options['wavenumber'])

    def _get_fmodel(self):
        """Return the line fitting model to use based on passed
        options.
        """
        fmodel = 'sincgauss'    

        if 'fmodel' in self.options:
            fmodel = self.options['fmodel']

        return fmodel

    def _get_fix_fwhm(self):
        """Return True if FWHM must be fixed."""
        if self._get_fmodel() == 'sincgauss':
            return True
        else:
            return False
            
    def fit_lines_maps(self):
        """fit lines maps.

        This is a wrapper around
        :py:meth:`orcs.SpectralCubeFitter.fit_lines_maps`.

        .. seealso:: :py:meth:`orcs.SpectralCubeFitter.fit_lines_maps`
        """       
            
        if 'signal_range' in self.options:
            signal_range = self.options['signal_range']
        else:
            signal_range = self.options['filter_range']

        self.spectralcube.fit_lines_maps(
            self.options['object_regions_path'],
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            signal_range,
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            axis_corr=self.options['axis_corr'],
            poly_order=self.options.get('poly_order'),
            sky_regions_file_path=self.options.get('sky_regions_path'),
            fmodel=self._get_fmodel(),
            fix_fwhm=self._get_fix_fwhm(),
            cov_pos=self.cov_pos,
            cov_sigma=self.cov_sigma,
            binning=self.options['binning'],
            apodization=float(self.options['apodization']),
            velocity_range=self.options['velocity_range'])


    def get_sky_radial_velocity(self):
        """Return sky radial velocity

        This is a wrapper around
        :py:meth:`orcs.SpectralCubeFitter.get_sky_radial_velocity`.

        .. seealso:: :py:meth:`orcs.SpectralCubeFitter.get_sky_radial_velocity`
        """
        
        self.spectralcube.get_sky_radial_velocity(
            self.options.get('sky_regions_path'),
            self.get_lines_fwhm(),
            self.options['filter_range'],
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            poly_order=self.options.get('poly_order'),
            fmodel=self._get_fmodel(),
            fix_fwhm=self._get_fix_fwhm(),
            apodization=float(self.options['apodization']))
        
    def extract_raw_lines_maps(self):
        """Extract raw lines maps.

        This is a wrapper around
        :py:meth:`orcs.SpectralCubeFitter.extract_raw_lines_maps`.

        .. seealso:: :py:meth:`orcs.SpectralCubeFitter.extract_raw_lines_maps`
        """
        self.spectralcube.extract_raw_lines_maps(
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            self.options['step'],
            self.options['order'],
            self.options['wavenumber'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'])
            
    def fit_integrated_spectra(self, verbose=True):
        """Fit integrated spectra.

        :param verbose: (Optional) If True, print the fit results
          (default True).

        This is a wrapper around
        :py:meth:`orcs.SpectralCubeFitter.fit_integrated_spectra`.

        .. seealso:: :py:meth:`orcs.SpectralCubeFitter.fit_integrated_spectra`

        """
        return self.spectralcube.fit_integrated_spectra(
            self.options['integ_reg_path'],
            self.options['lines'],
            self.options['mean_velocity'],
            self.get_lines_fwhm(),
            self.options['filter_range'],
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            poly_order=self.options.get('poly_order'),
            sky_regions_file_path=self.options.get('sky_regions_path'),
            fmodel=self._get_fmodel(),
            fix_fwhm=self._get_fix_fwhm(),
            cov_pos=self.cov_pos,
            cov_sigma=self.cov_sigma,
            plot=self.options['plot'],
            auto_sky_extraction=self.options['auto_sky_extraction'],
            sky_size_coeff=self.options['sky_size_coeff'],
            axis_corr=self.options['axis_corr'],
            apodization=float(self.options['apodization']),
            verbose=verbose,
            velocity_range=self.options['velocity_range'])

    def _get_temp_reg_path(self):
        """Return path to a temporary region file"""
        return self._get_data_prefix() + 'temp.reg'

    def _get_skymap_file_path(self):
        """Return path to the sky map file containing the results of
        the fit.
        """
        return self._get_data_prefix() + 'skymap.txt'

    def _get_skymap_fits_path(self):
        """Return path to the sky map file containing the interpolated
        sky velocity map.
        """
        return self._get_data_prefix() + 'skymap.fits'


    def fit_integrated_spectrum(self, x, y, r, plot=True):
        """Extract and fit the integrated spectrum of a given region.

        :param x: X position of the region
        
        :param y: Y position of the region
        
        :param r: radius of the region
        
        :param plot: (Optional) If True, plot the spectrum and its fit
          (default True).
        """
        integ_file_path = self._get_temp_reg_path()
        with self.open_file(integ_file_path, 'w') as f:
            f.write('circle({:.5f},{:.5f},{:.5f})\n'.format(
                x, y, r))
            
        self.options['integ_reg_path'] = integ_file_path
        self.options['plot'] = plot
        self.fit_integrated_spectra()

    def map_sky_velocity(self, div_nb=20, plot=True, x_range=None, y_range=None,
                         exclude_reg_file_path=None):
        """Map the sky velocity on rectangular grid and interpolate it
        to return a map of the velocity zero point that can be
        subtracted to the returned velocity map of the cube fit.

        :param div_nb: (Optional) Number of division on one axis of
          the rectangular grid. The total number of points is div_nb^2
          (default 15).

        :param plot: (Optional) If True, plot the result (default
          True)
        """
        if div_nb < 2:
            self._print_error('div_nb must be >= 2')
        MAX_R = 150
        integ_file_path = self._get_temp_reg_path()
        dimx = self.spectralcube.dimx
        dimy = self.spectralcube.dimy
        regions = list()
        
        if x_range is not None:
            xmin = int(np.min(x_range))
            xmax = int(np.max(x_range))
        else:
            xmin = 0 ; xmax = dimx
        if y_range is not None:
            ymin = int(np.min(y_range))
            ymax = int(np.max(y_range))
        else:
            ymin = 0 ; ymax = dimy
            
        self._print_msg('X range: {} {}, Y range: {} {}'.format(xmin, xmax, ymin, ymax))

        r = min((xmax - xmin), (ymax - ymin)) / float(div_nb + 2) / 4.
        r = min(r, MAX_R)
        self._print_msg('Radius: {}'.format(r))

        exclude_mask = np.zeros((dimx, dimy), dtype=bool)
        if exclude_reg_file_path is not None:
            exclude_mask[orb.utils.misc.get_mask_from_ds9_region_file(
                exclude_reg_file_path,
                [0, dimx], [0, dimy])] = True

            
        ## with open(integ_file_path, 'w') as f:
        ##     for ix in np.linspace(xmin, xmax, div_nb + 2)[1:-1]:
        ##         for iy in np.linspace(ymin, ymax, div_nb + 2)[1:-1]:
        ##             if not exclude_mask[ix, iy]:
        ##                 regions.append((ix, iy, r))
        ##                 f.write('circle({:.5f},{:.5f},{:.5f})\n'.format(
        ##                     ix, iy, r))

        ## self._print_msg('{} regions to fit'.format(len(regions)))
            
        ## self.options['integ_reg_path'] = integ_file_path
        ## self.options['plot'] = False
        ## self.options['sky_regions_path'] = None
        ## lines_nb = len(self.options['lines'])

        ## # fit sky spectra
        ## paramsfile = self.fit_integrated_spectra(verbose=False)

        ## # write results
        ## with open(self._get_skymap_file_path(), 'w') as f:
        ##     for ireg in range(len(regions)):
        ##         iv = paramsfile[ireg*lines_nb]['v']
        ##         iv_err = paramsfile[ireg*lines_nb]['v_err']
        ##         f.write('{} {} {} {}\n'.format(
        ##             regions[ireg][0], regions[ireg][1], iv, iv_err))

        # create map
        with open(self._get_skymap_file_path(), 'r') as f:
            vel = list()
            vel_err = list()
            x = list()
            y = list()
            for line in f:
                line = np.array(line.strip().split(), dtype=float)
                x.append(line[0])
                y.append(line[1])
                vel.append(line[2])
                vel_err.append(line[3])

        x = np.array(x)
        y = np.array(y)
        vel = np.array(vel)
        nans = np.isnan(vel)
        vel[nans] = 0.
        vel_err = np.array(vel_err)
        vel_err[vel_err == 0.] = np.nan

        vel_err[vel_err > (
            np.nanmedian(vel_err) + 2.5 * np.std(
                orb.utils.stats.sigmacut(
                    vel_err, sigma=2.)))] = np.nan

        # create weights map
        w = 1./(vel_err)
        #w /= np.nanmax(w)
        #w[np.isnan(w)] = 1e-35
        x = x[~np.isnan(w)]
        y = y[~np.isnan(w)]
        vel = vel[~np.isnan(w)]
        w = w[~np.isnan(w)]

        vel[vel < np.nanpercentile(vel, 16)] = np.nan
        vel[vel > np.nanpercentile(vel, 84)] = np.nan
        x = x[~np.isnan(vel)]
        y = y[~np.isnan(vel)]
        w = w[~np.isnan(vel)]

        vel = vel[~np.isnan(vel)]


        import pylab as pl
        pl.scatter(x,y,c=vel)
        pl.show()
        quit()



        # interpolate map
        s = None
        k = 3
        spl = scipy.interpolate.SmoothBivariateSpline(x, y, vel, w=w, s=s, kx=k, ky=k)
        Z = spl(np.arange(dimx), np.arange(dimy))
        vel[nans] = np.nan
        Z[:int(np.nanmin(x[~np.isnan(vel)])),:] = np.nan
        Z[int(np.nanmax(x[~np.isnan(vel)])):,:] = np.nan
        Z[:,:int(np.nanmin(y[~np.isnan(vel)]))] = np.nan
        Z[:,int(np.nanmax(y[~np.isnan(vel)])):] = np.nan

        # write map
        self.write_fits(
            self._get_skymap_fits_path(),
            Z, overwrite=True)
        
        # plot map
        if plot:
            import pylab as pl
            ## # remove mean
            ## vel -= np.nanmean(Z)
            ## Z -= np.nanmean(Z)

            vmin = orb.cutils.part_value(Z.flatten(), 0.03)
            vmax = orb.cutils.part_value(Z.flatten(), 0.97)
            
            pl.scatter(x, y, c=vel, vmin=vmin, vmax=vmax)
            pl.imshow(Z.T, vmin=vmin, vmax=vmax)
            pl.xlim((0, dimx))
            pl.ylim((0, dimy))
            pl.colorbar()
            cs = pl.contour(np.arange(dimx), np.arange(dimy),
                            Z.T, 10, colors='1.', linewidths=3.)
            pl.clabel(cs)
            pl.show()
    
        


