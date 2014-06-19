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
import bottleneck as bn

# import ORB
try:
    from orb.core import Tools, OptionFile, Lines, ProgressBar, ParamsFile
    import orb.utils
    
except IOError, e:
    print "ORB could not be found !"
    print e
    sys.exit(2)

from rvcorrect import RVCorrect
import utils




#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCube(Tools):

    data = None # Spectral cube data
    hdr = None # Spectral cube header
    nm_axis = None # Wavelength axis (from spectral cube header)
    nm_min = None
    nm_step = None

    object_name = None
    filter_name = None
    apodization_function = None
    
    dimx = None
    dimy = None
    dimz = None

    x_range = None
    y_range = None

    lines_nm = None # Vacuum wavelength (in nm) of the searched lines
    lines_shift = None # Wavelength shift of the lines (in nm)
    lines_fwhm = None # Rough FWHM (in nm) of the lines
    filter_min = None # Minimum wavelength (in nm) of the filter band
    filter_max = None # Maximum wavelength (in nm) of the filter band

    options = None
    overwrite = None

    lines = None
    
    def __init__(self, option_file_path, object_name=None,
                 cube_path=None, overwrite=True, no_data_load=False,
                 silent=True):

        # Load options file
        self.options = OptionFile(option_file_path)
        self.target_ra = self.options.get('TARGET_RA')
        self.target_dec = self.options.get('TARGET_DEC')
        self.obs_coords = [self.options.get('OBS_LAT', float),
                           self.options.get('OBS_LONG', float),
                           self.options.get('OBS_ALT', float)]
        self.date = self.options.get('DATE')
        if self.date != None:
            self.date = np.array(self.date.split(',')).astype(int)
        self.hour_ut = self.options.get('HOUR_UT')
        
        if cube_path == None:
            self.cube_path = self.options["CUBE_PATH"]
        else: self.cube_path = cube_path

        # Define searched lines and lines parameters
        lines = self.options.get_lines()
        if lines != None:
            if np.size(lines) == 1:
                lines = [lines]
            self.lines_nm = np.array(lines)
            
            if not silent:
                self._print_msg('Searched emission lines: %s'%str(self.lines_nm))
        else:
            self._print_error(
                "No searched emission line defined in the option file")
        
        lines_shift = self.options['SHIFT']
        if lines_shift != None:
            self.lines_shift = float(lines_shift)
        else:
            self.lines_shift = 0.

        if not silent:
            self._print_msg('Spectral shift: %f'%self.lines_shift)
        
        lines_fwhm = self.options['FWHM']
        if lines_fwhm != None:
            self.lines_fwhm = float(lines_fwhm)
        else:
            self._print_error("Lines FWHM must be defined. Check option file.")

        if not silent:
            self._print_msg('Lines rough FWHM: %f'%self.lines_fwhm)

        poly_order = self.options['POLY_ORDER']
        if poly_order != None:
            self.poly_order = int(poly_order)
        else:
            self.poly_order = 1.
        if not silent:
            self._print_msg('Order of the polynomial used to fit continuum: %d'%self.poly_order)
        filter_edges = self.options.get_filter_edges()
        if filter_edges != None:
           self.filter_min = np.min(filter_edges)
           self.filter_max = np.max(filter_edges)

        if not silent:
            self._print_msg("Filter edges: [%f,%f]"%(self.filter_min,
                                                     self.filter_max))
        
        if not no_data_load:
            # Load data
            self._print_msg("Loading %s"%self.cube_path, color=True)
            self.data, self.hdr = self.read_fits(
                self.cube_path, return_header=True)
            self.dimx = self.data.shape[0]
            self.dimy = self.data.shape[1]
            self.dimz = self.data.shape[2]
            if not silent:
                self._print_msg("Cube loaded : (%d, %d, %d)"%
                                (self.dimx, self.dimy, self.dimz))
            self.nm_axis = self._get_nm_axis()
        else:
            cube_hdu = self.read_fits(self.cube_path, return_hdu_only=True)
            self.hdr = cube_hdu[0].header

        # check lines positions
        if self.nm_axis != None and self.lines_nm != None:
            if np.any(self.lines_nm > np.max(self.nm_axis)):
                self._print_error('One of the lines is not in the spectrum')
            if np.any(self.lines_nm < np.min(self.nm_axis)):
                self._print_error('One of the lines is not in the spectrum')
            if np.any(self.lines_nm > self.filter_max):
                self._print_error('One of the lines is not in the filter')
            if np.any(self.lines_nm < self.filter_min):
                self._print_error('One of the lines is not in the filter')

        # Defining ROI
        self.x_range = [0, self.dimx]
        self.y_range = [0, self.dimy]
        if self.options['ROI'] != None:
            roi = np.array(self.options['ROI'].split(',')).astype(int)
            if roi.shape[0] == 4:
                self.x_range = [roi[0], roi[1]]
                self.y_range = [roi[2], roi[3]]
            else:
                self._print_warning('Bad ROI structure (Xmin,Xmax,Ymin,Ymax). Check option file.')

        # Define object name
        if object_name == None:
            self.object_name = self.hdr['OBJECT']
        else:
            self.object_name = object_name

        self.filter_name = self.hdr['FILTER']
        self.apodization_function = self.hdr['APODIZ']
        
        self._data_prefix = self._get_data_prefix()
        self._data_path_hdr = self._get_data_path_hdr()
        self.overwrite = overwrite
        self.lines = Lines()
        
    def _get_data_prefix(self):
        return (os.curdir + os.sep + self.object_name + '_' + self.filter_name
                + '_' + self.apodization_function
                + os.sep  + self.object_name + '_' + self.filter_name
                + '_' + self.apodization_function + '_')

    def _get_maps_list_path(self):
        """Return path to the list of extacted emission lines maps"""
        return self._data_path_hdr + "maps_list"

    def _get_fits_header(self, file_type, comment=None,
                         center_shift=None):
        new_hdr = self.hdr
        if comment == None: comment = 'Type of file'
        new_hdr['FILETYPE'] = (file_type, comment)
        if center_shift != None:
            if ('CRPIX1' in new_hdr.keys()
                and 'CRPIX2' in new_hdr.keys()):
                new_hdr['CRPIX1'] = new_hdr['CRPIX1'] + center_shift[0]
                new_hdr['CRPIX2'] = new_hdr['CRPIX2'] + center_shift[1]
                
        if 'PROGRAM' in new_hdr.keys():
            comment_pos = new_hdr.keys().index('PROGRAM') + 1
        else:
            comment_pos = -1
            
        new_hdr.insert(comment_pos, ('COMMENT', 'Generated by ORCS (Thomas Martin: thomas.martin.1@ulaval.ca'))
        return new_hdr
        

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
        
    def _get_sub_cube_path(self):
        return self._data_path_hdr + "sub.fits"

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
    
    def _get_binned_cube_path(self, binning):
        return self._data_path_hdr + "binned_%d.fits"%binning
    
    def _get_integrated_spectrum_path(self, region_name):
        return self._data_path_hdr + "integrated_spectrum_%s"%region_name
    
    def _get_integrated_spectrum_map_path(self, region_name):
        return self._data_path_hdr + "integrated_spectrum_map_%s"%region_name
    
    def _get_nm_axis(self):
        dimz = self.hdr["NAXIS3"]
        if dimz != self.dimz:
            self._print_error('Bad axis definition')
        
        self.nm_min = self.hdr["CRVAL3"]
        self.nm_step = self.hdr["CDELT3"]
        ref_point = self.hdr["CRPIX3"]
        return (np.arange(self.dimz) + ref_point - 1) * self.nm_step + self.nm_min

    def _get_signal_range(self):
        SIGNAL_BORDER_COEFF = 0.01
        signal_range = [orb.utils.nm2pix(self.nm_axis, self.filter_min),
                        orb.utils.nm2pix(self.nm_axis, self.filter_max)]
        signal_length = signal_range[-1] - signal_range[0]
        signal_border = int(math.ceil(SIGNAL_BORDER_COEFF * signal_length))
        signal_range[0] += signal_border
        signal_range[1] -= signal_border
        return np.array(signal_range).astype(int)

    def _get_fwhm_pix(self):
        """Return fwhm in channels from default lines FWHM"""
        return orb.utils.nm2pix(self.nm_axis,
                                self.nm_axis[0] + self.lines_fwhm)
    
    def _extract_spectrum_from_region(self, region,
                                      substract_spectrum=None,
                                      median=True, silent=False):
        """
        :param region: A list of the indices of the pixel integrated
          in the returned spectrum.
        
        :param lines: Rough wavelength in nm of the lines to be
          fitted, if None no fit is done on the extracted
          spectrum (default None).
          
        :param fwhm_guess: Initial guess on the lines FWHM (default
          3.5)

        :substract_spectrum: Remove the given spectrum from the
          extracted spectrum before fitting parameters. Useful
          to remove sky spectrum. Both spectra must have the same size.

        :param median: If True the integrated spectrum is the median
          of the spectra. Else the integrated spectrum is the mean of
          the spectra (Default False).

        :param silent: (Optional) If True, nothing is printed (default
          False).
        """
        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1
            
        spectrum = np.zeros(self.dimz)
        if not silent: progress = ProgressBar(self.dimz)
        for iframe in range(self.dimz):
            spectra = (self.data[:,:,iframe])[np.nonzero(mask)].flatten()
            if median:
                spectrum[iframe] = bn.nanmedian(spectra)
            else:
                spectrum[iframe] = bn.nanmean(spectra)
            if not silent: progress.update(iframe)
        if not silent: progress.end()

        if substract_spectrum != None:
            spectrum -= substract_spectrum
            
        return spectrum
        
    def _fit_lines_in_cube(self, cube, searched_lines=[],
                           detect_coeff=3., fwhm_guess=3.5,
                           signal_range=None, x_range=None,
                           y_range=None, fix_fwhm=False,
                           fmodel='gaussian', observation_params=None,
                           cov_pos=True, cov_fwhm=True):
        
        """Fit gaussian shaped lines in a spectral cube.

        :param cube: Spectral cube to fit
        
        :param searched_lines: lines channels.
        
        :param detect_coeff: (Optional) WARNING: deprecated. Detection
          coefficient. Detection threshold = median + detect_coeff *
          std (Default 3.).

        :param fwhm_guess: (Optional) Initial guess on the lines fwhm
          (default 3.5).

        :param signal_range: (Optional) Range of channels with signal
          (Generally the filter band pass) (Default None)

        :param x_range: (Optional) Range of pixels along the x axis
          where lines are fitted (Default None).
          
        :param y_range: (Optional) Range of pixels along the y axis
          where lines are fitted (Default None).

        :param fix_fwhm: (Optional) If True FWHM is fixed to
          fwhm_guess value (defautl False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param observation_params: (Optional) Must be a tuple [step,
          order]. Fits are better, because data is interpolated to
          remove distorsions due to the interpolation from an axis
          regular in cm-1 to an axis regular in nm during the FFT.
        """

        def _fit_lines_in_column(data, searched_lines,
                                 fwhm_guess, y_min, y_max,
                                 poly_order, fix_fwhm, fmodel,
                                 observation_params, signal_range,
                                 cov_pos, cov_fwhm):
            fit = np.empty((data.shape[0], len(searched_lines), 4),
                           dtype=float)
            fit.fill(np.nan)
            
            err = np.empty((data.shape[0], len(searched_lines), 4),
                           dtype=float)
            err.fill(np.nan)
            
            res = np.empty((data.shape[0], len(searched_lines), 2),
                           dtype=float)
            res.fill(np.nan)

            for ij in range(y_min, y_max):
                result_fit = orb.utils.fit_lines_in_vector(
                    data[ij,:],
                    searched_lines,
                    fwhm_guess=fwhm_guess,
                    cont_guess=None,
                    fix_cont=False,
                    poly_order=poly_order,
                    cov_pos=cov_pos,
                    cov_fwhm=cov_fwhm,
                    fix_fwhm=fix_fwhm,
                    fmodel=fmodel,
                    interpolation_params=observation_params,
                    signal_range=signal_range)
              
                for iline in range(len(searched_lines)):
                    if result_fit != []:
                        fit[ij,iline,:] = result_fit['lines-params'][iline]
                        err[ij,iline,:] = result_fit['lines-params-err'][iline]
                        res[ij,iline,:] = [result_fit['reduced-chi-square'],
                                           result_fit['snr'][iline]]
                            
                    else:
                        fit[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN')]
                        err[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN')]
                        res[ij,iline,:] = [float('NaN'), float('NaN')]

            return fit, err, res
                                
        if len(cube.shape) == 1:
            cube = cube.reshape((1,1,cube.shape[0]))
        
        ## Fit lines
        # Create arrays
        self._print_msg("fitting lines")
        if np.size(searched_lines) == 1:
            searched_lines = [searched_lines]
        hei_image = (np.empty((cube.shape[0],cube.shape[1],len(searched_lines)),
                              dtype=float))
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
        
        # Defining computation range
        if x_range != None:
            x_min = np.min(x_range)
            x_max = np.max(x_range)
            if x_min < 0:
                self._print_warning("x min must be >= 0, x min set to 0")
                x_min == 0
            if x_max > cube.shape[0]:
                self._print_warning("x max must be <= %d, x max set to %d"%(
                    cube.shape[0], cube.shape[0]))
                x_max = cube.shape[0]
        else:
            x_min = 0
            x_max = cube.shape[0]
        if y_range != None:
            y_min = np.min(y_range)
            y_max = np.max(y_range)
            if y_min < 0:
                self._print_warning("y min must be >= 0, y min set to 0")
                y_min == 0
            if y_max > cube.shape[1]:
                self._print_warning("y max must be <= %d, y max set to %d"%(
                    cube.shape[1], cube.shape[1]))
                y_max = cube.shape[1]
        else:
            y_min = 0
            y_max = cube.shape[1]
            
        # multi-processing server init
        job_server, ncpus = self._init_pp_server()
        
        progress = ProgressBar(x_max-x_min)
        for ii in range(x_min, x_max, ncpus):
            # no more jobs than columns
            if (ii + ncpus >= x_max): 
                ncpus = x_max - ii
                
            # jobs creation
            jobs = [(ijob, job_server.submit(
                _fit_lines_in_column,
                args=(cube[ii+ijob,:,:],
                      searched_lines,
                      fwhm_guess,
                      y_min, y_max,
                      self.poly_order,
                      fix_fwhm, fmodel,
                      observation_params,
                      signal_range, cov_pos, cov_fwhm), 
                modules=("import numpy as np", 
                         "import orb.utils",
                         "import orcs.utils as utils"),
                depfuncs=(orb.utils.fit_lines_in_vector,)))
                    for ijob in range(ncpus)]
            for ijob, job in jobs:
                (fit, err, res) = job()
                hei_image[ii+ijob,:,:] = fit[:,:,0]
                amp_image[ii+ijob,:,:] = fit[:,:,1]
                shi_image[ii+ijob,:,:] = fit[:,:,2]
                wid_image[ii+ijob,:,:] = fit[:,:,3]
                hei_image_err[ii+ijob,:,:] = err[:,:,0]
                amp_image_err[ii+ijob,:,:] = err[:,:,1]
                shi_image_err[ii+ijob,:,:] = err[:,:,2]
                wid_image_err[ii+ijob,:,:] = err[:,:,3]
                chisq_image[ii+ijob,:,:] = res[:,:,0]
                snr_image[ii+ijob,:,:] = res[:,:,1]
                
            progress.update(ii-x_min, info="column : %d/%d"%(ii-x_min,x_max-x_min))
        self._close_pp_server(job_server)
        progress.end()
    
        results_list = list()
        for iline in range(len(searched_lines)):
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

    def compute_radial_velocity_correction(self):
        """Return radial velocity correction
        """
        return RVCorrect(self.target_ra, self.target_dec, self.date,
                         self.hour_ut, self.obs_coords).rvcorrect()

    def create_deep_frame(self, write=True):
        """Create a deep frame from a cube.

        :param write: (Optional) Write the computing deep frame (default True).
        """
        deep_frame = np.nanmean(self.data, axis=2)
        if write:
            self.write_fits(self._get_deep_frame_path(), deep_frame,
                            overwrite=self.overwrite,
                            fits_header=self._get_fits_header("Deep Frame"))
        return deep_frame

    def create_sky_mask(self, sky_regions_file_path, threshold_coeff=0.3):
        """Create a mask from the pixels corresponding to the sky
        
        :param sky_regions_file_path: Path to a file containing the
          coordinates of the regions of the sky.

        .. note:: The region file can be created using ds9 and the
          region shape 'box'. The region file is then saved
          ('region/save regions') using the default parameters
          ('format : ds9', 'Coordinates system : physical').
        """
        sky_regions = orb.utils.get_mask_from_ds9_region_file(
            sky_regions_file_path,
            x_range=[0, self.dimx],
            y_range=[0, self.dimy])
        deep_frame = self.create_deep_frame(write=False)
        sky = deep_frame[sky_regions]
        # std cut to remove stars from the sky
        sky_median = np.median(sky)
        sky_std = np.std(sky)
        sky[np.nonzero(sky > sky_median + sky_std)] = 0.
        sky[np.nonzero(sky < 0)] = 0.
        sky_median = np.median(sky[np.nonzero(sky)])
        sky_std = np.std(sky[np.nonzero(sky)])
        
        mask_map = np.ones_like(deep_frame)
        mask_map[np.nonzero(
            deep_frame < sky_median + threshold_coeff * sky_std)] = 0

        self.write_fits(self._get_sky_mask_path(), mask_map,
                        overwrite=self.overwrite,
                        fits_header=self._get_fits_header("Sky Mask"))
        return mask_map
    
    def create_sub_cube(self, x_min, x_max, y_min, y_max):
        sub_data = self.data[x_min:x_max, y_min:y_max]
        self.write_fits(self._get_sub_cube_path(), sub_data,
                        fits_header=self._get_fits_header(
                            "Sub Cube",
                            center_shift=[-x_min, -y_min]),
                        overwrite=self.overwrite)
        return sub_data

    def create_binned_cube(self, binning=2):
        binned_cube = np.empty((int(self.dimx/binning),
                                int(self.dimy/binning),
                                self.dimz))
        progress = ProgressBar(binned_cube.shape[0])
        for ii in range(binned_cube.shape[0]):
            progress.update(ii, info="Binning")
            for ij in range(binned_cube.shape[1]):
                binned_pixel = self.data[int(binning*ii):
                                         int(binning*(ii+1)),
                                         int(binning*ij):
                                         int(binning*(ij+1)),:]
                binned_pixel = np.mean(np.mean(binned_pixel, axis=0), axis=0)
                binned_cube[ii,ij,:] = binned_pixel
        progress.end()
        
        self.write_fits(self._get_binned_cube_path(binning),
                        binned_cube, overwrite=self.overwrite,
                        fits_header=self._get_fits_header("Binned Cube"))
        return binned_cube

    def get_sky_radial_velocity(self, sky_region_file_path, fmodel='gaussian',
                                lines_shift=0., show=True):
        """
        .. note:: Use orb.Lines() and Ostebrock et al., PASP, 108,
          277 (1996) data to get the sky lines to fit.

            ================ =========================
            OH Line          Air wavelength (in nm)
            ================ =========================
            6-1 Q1(1.5)      649.8729
            6-1 P1(2.5)      653.3044
            6-1 P1(3.5)      655.3617
            6-1 P1e,f(4.5)   657.72845 (doublet)
            6-1 P1e,f(5.5)   660.41345 (doublet)
    
        """
        # compute signal range
        signal_range = self._get_signal_range().astype(int)
        
        sky_lines_nm = Lines().get_sky_lines(self.filter_min, self.filter_max,
                                             self.lines_fwhm)
        
        sky_lines_nm_shift = np.array(sky_lines_nm) + lines_shift
        sky_lines_pix = orb.utils.nm2pix(self.nm_axis, sky_lines_nm_shift)
        
   
        region = orb.utils.get_mask_from_ds9_region_file(
            sky_region_file_path,
            x_range=[0, self.dimx], y_range=[0, self.dimy])
        sky_median_spectrum = self._extract_spectrum_from_region(
            region, median=True)

        cont_guess = orb.utils.robust_median(sky_median_spectrum)
        fix_cont = False
        

        fit = orb.utils.fit_lines_in_vector(
            sky_median_spectrum,
            np.copy(sky_lines_pix),
            cont_guess=[cont_guess],
            fix_cont=fix_cont,
            poly_order=0,
            fmodel=fmodel,
            cov_pos=True,
            cov_fwhm=True,
            return_fitted_vector=True,
            fwhm_guess=self._get_fwhm_pix(),
            fix_fwhm=False,
            signal_range=signal_range,
            interpolation_params=[float(self.hdr['STEP']),
                                  float(self.hdr['ORDER'])])
    
        fit_params = fit['lines-params']
        fit_params_err = fit['lines-params-err']
        
            
        velocity = list()
        for iline in range(len(fit_params)):
            pline = fit_params[iline]
            velocity.append(utils.compute_radial_velocity(
                orb.utils.pix2nm(self.nm_axis, pline[2]),
                sky_lines_nm[iline]))
            
            err = utils.compute_radial_velocity(
                orb.utils.pix2nm(self.nm_axis, fit_params_err[iline][2]
                 + sky_lines_pix[iline]),
                orb.utils.pix2nm(self.nm_axis, sky_lines_pix[iline]))
            
            self._print_msg("Line %f nm, velocity : %f km/s [+/-] %f"%(
                sky_lines_nm[iline], velocity[-1], err))
            
        self._print_msg(
            "Mean sky radial velocity : %f km/s"%np.mean(np.array(velocity)))

        if show:
            import pylab as pl
            for iline in range(np.size(sky_lines_nm_shift)):
                pl.axvline(x=sky_lines_nm_shift[iline], alpha=0.5,
                           color='0.', linestyle=':')
            pl.plot(self.nm_axis, sky_median_spectrum, color='0.5',
                    label = 'sky spectrum')
            pl.plot(self.nm_axis, fit['fitted-vector'], color='0.',
                    linewidth=2., label='fit')
            pl.legend()
            pl.show()

        err_params = np.nan
        return sky_median_spectrum, fit_params, err_params
        
            
    def extract_lines_maps(self, sky_regions_file_path=None,
                           roi_file_path=None, cov_pos=True, cov_fwhm=True,
                           detect_coeff=0., fixed_fwhm_value=None,
                           fmodel='gaussian', observation_params=None):
        
        """
        Extract emission lines parameters maps from a fit.

        All parameters of each emission line are mapped. So that for a
        gaussian fit with 4 parameters (height, amplitude, shift,
        fwhm) 4 maps are created for each fitted emission line in the
        cube.

        :sky_regions_file_path: Path to a ds9 region file giving the
          pixels where the sky spectrum has to be extracted. This
          spectrum will be substracted to the spectral cube before it
          is fitted.

        :roi_file_path: Path to a ds9 region file giving the regions
          of interest to extract.

        :param detect_coeff: Define the detection threshold from the
          computed noise in number of standard deviation over the median counts.

        :param fix_fwhm_value: Fixed value of the FWHM in nm. Might be
          useful after a first extraction to reextract data using the
          same FWHM everywhere (default None).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param observation_params: (Optional) Must be a tuple [step,
          order]. Fits are better, because data is interpolated to
          remove distorsions due to the interpolation from an axis
          regular in cm-1 to an axis regular in nm during the FFT.
        """
        self._print_msg("Extracting lines data", color=True)

        rest_frame_lines_nm = np.copy(self.lines_nm)
        lines_nm = self.lines_nm + self.lines_shift 

        # Extract median sky spectrum
        if sky_regions_file_path != None:
            self._print_msg("Extracting sky median vector")
            median_sky_spectrum = self._extract_spectrum_from_region(
                orb.utils.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy]))

            data = self.data - median_sky_spectrum
        else:
            data = self.data

        
       # remove non-ROI pixels
        if roi_file_path is not None:

            roi_mask = np.zeros((self.dimx, self.dimy), dtype=bool)
            roi_mask[orb.utils.get_mask_from_ds9_region_file(
                roi_file_path,
                x_range=[0, self.dimx],
                y_range=[0, self.dimy])] = True
           
            data = np.copy(self.data)
            for ii in range(self.dimx):
                for ij in range(self.dimy):
                    if not roi_mask[ii,ij]:
                        data[ii,ij,:] = np.nan
                        
        else:
            data = self.data
            
        # compute signal range
        signal_range = self._get_signal_range()

        # fixed fwhm
        if fixed_fwhm_value != None:
            fwhm_guess = orb.utils.nm2pix(self.nm_axis,
                                self.nm_axis[0] + fixed_fwhm_value)
            self._print_warning('FWHM is fixed to %f nm [%f pix]'%(
                fixed_fwhm_value, fwhm_guess))
            fix_fwhm = True
        else:
            fwhm_guess = self._get_fwhm_pix()
            fix_fwhm = False


        self._print_error('Attention, il faut avant tout revor la routine de fit qui a changé depuis')
        results = self._fit_lines_in_cube(
            data,
            searched_lines = orb.utils.nm2pix(
                self.nm_axis, lines_nm),
            signal_range=signal_range,
            x_range=self.x_range,
            y_range=self.y_range,
            fwhm_guess=fwhm_guess,
            detect_coeff=detect_coeff,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            observation_params=observation_params,
            cov_pos=cov_pos, cov_fwhm=cov_fwhm)

        # SAVE MAPS
        maps_list_path = self._get_maps_list_path()
        maps_list = self.open_file(maps_list_path, 'w')

        for iline in range(np.size(lines_nm)):    
            line_name = Lines().round_nm2ang(rest_frame_lines_nm[iline])
            
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
            
            # write FWHM map in nm
            map_path = self._get_map_path(line_name,
                                          param='fwhm')
            fwhm_map = results[iline][3] * self.nm_step
            self.write_fits(
                map_path, fwhm_map,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "FWHM Map %d"%line_name,
                    comment='in nm'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'FWHM'))

            # write FWHM map in nm
            map_path = self._get_map_path(line_name,
                                          param='fwhm',
                                          err=True)
            fwhm_map = results[iline][7] * self.nm_step
            self.write_fits(
                map_path, fwhm_map,
                overwrite=self.overwrite,
                fits_header=self._get_fits_header(
                    "FWHM Map Error %d"%line_name,
                    comment='in nm'))
            maps_list.write(map_path + ' %d %s\n'%(line_name, 'FWHM_ERR'))


            # write velocity map
            # compute radial velocity from shift
            shift_nm = orb.utils.pix2nm(self.nm_axis, results[iline][2])

            velocity = utils.compute_radial_velocity(
                shift_nm,
                rest_frame_lines_nm[iline])
            
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
            shift_nm = orb.utils.pix2nm(self.nm_axis, results[iline][6])

            velocity = utils.compute_radial_velocity(
                shift_nm, self.nm_axis[0])

            
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
                

    def extract_integrated_spectra(self, regions_file_path,
                                   sky_regions_file_path=None,
                                   median=False, poly_order=None,
                                   observation_params=None,
                                   signal_range=None, plot=True,
                                   auto_sky_extraction=True,
                                   sky_size_coeff=2.):
        
        if poly_order is None:
            poly_order = self.poly_order


        # Create parameters file
        paramsfile = orb.core.ParamsFile(
            self._get_integrated_spectra_fit_params_path())
        
        # Extract sky spectrum
        if sky_regions_file_path != None:
            self._print_msg("Extracting sky median vector")
            reg_mask = orb.utils.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    x_range=[0, self.dimx],
                    y_range=[0, self.dimy])

            median_sky_spectrum = self._extract_spectrum_from_region(
                reg_mask)
        else: median_sky_spectrum = None
            

        # extract regions
        integ_spectra = list()
        with open(regions_file_path, 'r') as f:
            region_index = 0
            for ireg in f:
                if len(ireg) > 3 and '#' not in ireg:
                    region = orb.utils.get_mask_from_ds9_region_line(
                        ireg,
                        x_range=[0, self.dimx],
                        y_range=[0, self.dimy])
                    if len(region[0]) > 0:
                        if auto_sky_extraction:
                            # get sky around region
                            reg_bary = (np.mean(region[0]),
                                        np.mean(region[1]))
                            reg_size = max(np.max(region[0])
                                           - np.min(region[0]),
                                           np.max(region[1])
                                           - np.min(region[1]))

                            sky_size = reg_size * sky_size_coeff
                            x_min, x_max, y_min, y_max = (
                                orb.utils.get_box_coords(
                                    reg_bary[0], reg_bary[1], sky_size,
                                    0, self.dimx, 0, self.dimy))

                            x_sky = list()
                            y_sky = list()
                            for ipix in range(int(x_min), int(x_max)):
                                for jpix in range(int(y_min), int(y_max)):
                                    if ((math.sqrt((ipix - reg_bary[0])**2.
                                                   + (jpix - reg_bary[1])**2.)
                                         <= round(sky_size / 2.))):
                                        x_sky.append(ipix)
                                        y_sky.append(jpix)

                            sky = list([x_sky, y_sky])
                            mask = np.zeros((self.dimx, self.dimy))
                            mask[sky] = 1
                            mask[region] = 0
                            sky = np.nonzero(mask)
                            
                            median_sky_spectrum = (
                                self._extract_spectrum_from_region(
                                    sky, median=True, silent=True))
                            
                        # extract spectrum
                        spectrum = self._extract_spectrum_from_region(
                            region, substract_spectrum=median_sky_spectrum,
                            median=median, silent=True)

                        pix_center = int(self.nm_axis.shape[0]/2.)
                        shift_guess = (
                            orb.utils.nm2pix(
                                self.nm_axis,
                                self.nm_axis[pix_center] + self.lines_shift)
                            - pix_center)

                        # fit spectrum
                        fit = orb.utils.fit_lines_in_vector(
                            spectrum,
                            orb.utils.nm2pix(self.nm_axis, self.lines_nm),
                            fwhm_guess=self._get_fwhm_pix(),
                            cont_guess=None,
                            shift_guess=shift_guess,
                            fix_cont=False,
                            poly_order=poly_order,
                            cov_pos=True,
                            cov_fwhm=True,
                            fix_fwhm=False,
                            fmodel='gaussian',
                            interpolation_params=observation_params,
                            signal_range=self._get_signal_range(),
                            return_fitted_vector=True)

                        if fit != []:
                            fit_params = fit['lines-params']
                            err_params = fit['lines-params-err']
                            real_lines_nm = orb.utils.pix2nm(
                                self.nm_axis, fit_params[:, 2])

                            velocities = utils.compute_radial_velocity(
                                real_lines_nm,
                                self.lines_nm)

                            err_lines_nm = orb.utils.pix2nm(
                                self.nm_axis, err_params[:, 2])

                            velocities_err = utils.compute_radial_velocity(
                                err_lines_nm, self.nm_axis[0])
                            
                            for iline in range(fit_params.shape[0]):
                                snr = abs(fit_params[iline, 1]
                                          / err_params[iline, 1])
                                fit_results = {
                                    'reg_index': region_index,
                                    'line_name': self.lines.get_line_name(
                                        self.lines_nm[iline]),
                                    'h': fit_params[iline, 0],
                                    'a': fit_params[iline, 1],
                                    'v': velocities[iline],
                                    'fwhm': fit_params[iline, 3],
                                    'h_err': err_params[iline, 0],
                                    'a_err': err_params[iline, 1],
                                    'v_err': velocities_err[iline],
                                    'fwhm_err': err_params[iline, 3],
                                    'snr': snr}

                                paramsfile.append(fit_results)
                                print paramsfile[-1]

                        if plot:
                            import pylab as pl
                            pl.plot(self.nm_axis, spectrum,
                                    label='orig spectrum')
                            if fit != []:
                                pl.plot(self.nm_axis, fit['fitted-vector'],
                                        label='fit')
                            pl.legend()
                            pl.show()
                            
                        integ_spectra.append(spectrum)
                        
                region_index += 1
                    
        return integ_spectra
        

    def get_fitted_cube(self, x_range=None, y_range=None):

        self._print_error('Not implemented yet')

        self._print_msg("Extracting lines data", color=True)
            
        #rest_frame_lines_nm = np.copy(self.lines_nm)
        lines_nm = self.lines_nm + self.lines_shift
        
        if x_range == None:
            x_range = self.x_range
        if y_range == None:
            y_range = self.y_range
            
        results = self._fit_lines_in_cube(
            self.data,
            searched_lines = orb.utils.nm2pix(
                self.nm_axis, lines_nm),
            signal_range=[orb.utils.nm2pix(self.nm_axis,
                                           self.filter_min),
                          orb.utils.nm2pix(self.nm_axis,
                                           self.filter_max)],
            x_range=x_range,
            y_range=y_range,
            fwhm_guess=int(self._get_fwhm_pix()),
            detect_coeff=0.,
            fix_cont=False)

        x_min = np.min(x_range)
        x_max = np.max(x_range)
        y_min = np.min(y_range)
        y_max = np.max(y_range)

        fitted_cube = np.empty((x_max - x_min, y_max - y_min, self.dimz),
                               dtype=float)
        
        for ii in range(x_max - x_min):
            for ij in range(y_max - y_min):
                fitted_spectrum = np.zeros(self.dimz, dtype=float)
                for iline in range(len(results)):
                    amp = results[iline][1][ii + x_min,ij + y_min]
                    dx = results[iline][2][ii + x_min,ij + y_min]
                    fwhm = results[iline][3][ii + x_min,ij + y_min]
                    
                    ## fit_min = int(dx - 3 * fwhm)
                    ## fit_max = int(dx + 3 * fwhm + 1)
                    ## if fit_min < 0: fit_min = 0
                    ## if fit_max >= self.dimz: fit_max = self.dimz - 1
                    gauss_line = orb.utils.gaussian1d(
                        np.arange(self.dimz),
                        0., amp, dx, fwhm)
                    fitted_spectrum += gauss_line
                    
                fitted_cube[ii,ij,:] = fitted_spectrum
                    
        return fitted_cube

    def extract_raw_lines_maps(self):
        """Raw extraction of a portion of the spectral cube. Return
        the median of the slice.

        :param line_nm: Wavelength in nm of the line to extract. Can
          be a float or a string recognized by ORB (see
          :py:class:`orb.core.Lines`)

        .. note: Shift in the option file is taken into account
        """
        lines_nm = self.lines_nm + self.lines_shift
        lines_pix = orb.utils.nm2pix(self.nm_axis, lines_nm)
        
        fwhm_pix = self._get_fwhm_pix()
        min_pix = np.floor(lines_pix - fwhm_pix)
        min_pix[np.nonzero(min_pix < 0)] = 0
        max_pix = np.ceil(lines_pix + fwhm_pix) + 1
        max_pix[np.nonzero(max_pix >= self.dimz)] = self.dimz - 1

        # get continuum map
        mask_vector = np.ones(self.dimz, dtype=float)
        for iline in range(len(lines_pix)):
            mask_vector[min_pix[iline]:max_pix[iline]] = 0

        cont_map = np.nanmean(np.squeeze(
            self.data[:,:,np.nonzero(mask_vector)]), axis=2)

        cont_map_std = np.nanstd(np.squeeze(
            self.data[:,:,np.nonzero(mask_vector)]), axis=2)
        
        # write maps
        for iline in range(len(lines_pix)):
            line_map = (bn.nanmean(
                self.data[:,:,min_pix[iline]:max_pix[iline]], axis=2)
                        - cont_map)
            err_map = cont_map_std / math.sqrt(max_pix[iline] - min_pix[iline])
    
            raw_path = self._get_raw_line_map_path(
                int(round(self.lines_nm[iline]*10.)))
            raw_header = self._get_fits_header(
                    'RAW LINE MAP {:d}'.format(int(round(
                    self.lines_nm[iline]*10.))))

            err_path = self._get_raw_err_map_path(
                int(round(self.lines_nm[iline]*10.)))
            err_header = self._get_fits_header(
                    'RAW ERR MAP {:d}'.format(int(round(
                    self.lines_nm[iline]*10.))))
            
            self.write_fits(raw_path, line_map, overwrite=self.overwrite,
                fits_header=raw_header)
            self.write_fits(err_path, err_map, overwrite=self.overwrite,
                fits_header=err_header)
        
        
        
        
