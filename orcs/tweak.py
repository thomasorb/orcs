#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: tweak.py

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
import bottleneck as bn
import warnings

# import core
from core import HDFCube, Tools, OrcsBase
from process import SpectralCubeTweaker

##################################################
#### CLASS OrcsTeak ##############################
##################################################

class OrcsTweak(OrcsBase):


    def __init__(self, spectrum_cube_path, **kwargs):
        """Init class

        :param spectrum_cube_path: Path to the spectral cube. 
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties. 
        """
        OrcsBase.__init__(self, spectrum_cube_path, **kwargs)
        self.spectralcube = SpectralCubeTweaker(
            self.options['spectrum_cube_path'],
            data_prefix=self._get_data_prefix(),
            project_header=self._get_project_fits_header(),
            wcs_header = self.wcs_header,
            overwrite=self.overwrite,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus)


    def extract_integrated_spectrum(self, ds9_region_file_path):
        """Extract the integrated sky spectrum
        
        :param plot: (Optional) If True, plot the spectrum and its fit
          (default True).
        """

        median_spectrum = self.spectralcube.extract_integrated_spectrum(
            ds9_region_file_path,
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            axis_corr=self.options['axis_corr'])


    def subtract_spectrum(self, spectrum_file_path):

        spectrum = self.read_fits(spectrum_file_path)
        axis = None

        self.spectralcube.subtract_spectrum(
            spectrum, None,
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            axis_corr=self.options['axis_corr'])
        

    def detect_sources(self, signal_range=None, fast=True):

        if signal_range is None:
            signal_range = self.options['filter_range']

        self.spectralcube.detect_sources(
            self.options['wavenumber'],
            self.options['step'],
            self.options['order'],
            self.options['calibration_laser_map_path'],
            self.options['wavelength_calibration'],
            self.config['CALIB_NM_LASER'],
            signal_range,
            axis_corr=self.options['axis_corr'],
            fast=fast)
        
