#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: process.py

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

This module contains the processing classes
"""

import version
__version__ = version.__version__

# import Python libraries
import os
import sys
import math
import time
import numpy as np
import astropy.io.fits as pyfits
import bottleneck as bn
import warnings
import inspect
import scipy.interpolate
import marshal

# import core
from core import HDFCube, LineMaps, Tools

# import ORB
import orb.core
import orb.utils.spectrum
import orb.utils.image
import orb.utils.stats
import orb.utils.filters
import orb.utils.misc
import orb.fit
    
        
#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCubeFitter(HDFCube):

    """ORCS spectral cube fitting class.

    .. note:: parent class HDFCube is the ORCS implementation of
      HDFCube.
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

            if 'RESTFRQ' in new_hdr: del new_hdr['RESTFRQ']
            if 'RESTWAV' in new_hdr: del new_hdr['RESTWAV']
            if 'LONPOLE' in new_hdr: del new_hdr['LONPOLE']
            if 'LATPOLE' in new_hdr: del new_hdr['LATPOLE']

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
    
    def _get_integrated_spectrum_fit_path(self, region_name):
        dirname = os.path.dirname(self._data_path_hdr)
        basename = os.path.basename(self._data_path_hdr)
        return (dirname + os.sep + "INTEGRATED"
                + os.sep + basename + "integrated_spectrum_fit_{}.fits".format(region_name))

        
    def _fit_lines_in_cube(self, region, lines, step, order,
                           fwhm_guess, wavenumber,
                           calibration_coeff_map,
                           calibration_laser_map,
                           nm_laser,
                           poly_order, filter_range, shift_guess,
                           fix_fwhm=False, fmodel='gaussian',
                           subtract=None, cov_pos=True, cov_sigma=True,
                           cov_fwhm=True, binning=1,
                           apodization=None, velocity_range=None):
        
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

        :param calibration_coeff_map: Calibration coeff map
          (calibration laser map/ laser wavelength). Used to compute
          the line position in pixels.
          
        :param calibration_laser_map: Calibration laser map (must
          always be the real one to compute the real fwhm)
        
        :param nm_laser: Laser wavelength in nm.  
    
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
          are shifted with a covarying value.
    
        :param cov_pos: (Optional) Lines sigma in each spectrum
          have a covarying value.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.

        :param binning: (Optional) On the fly data binning (default 1).

        :param apodization: (Optional) Apodization level of the
          spectra. Line broadening due to apodization is
          removed from the fitting results (default None).

        :param velocity_range: (Optional) A float setting the range of
          velocities in km/s where the lines can be found (e.g. for a
          galaxy, default None)
        """
        def _fit_lines_in_column(data, coeff_map_col,
                                 calib_map_col, nm_laser, mask_col,
                                 lines, fwhm_guess, filter_range,
                                 poly_order, fix_fwhm, fmodel,
                                 step, order, wavenumber,
                                 cov_pos, cov_fwhm, subtract,
                                 shift_guess, cov_sigma, binning,
                                 apodization, velocity_range,
                                 init_velocity_map_col, init_sigma_map_col):

            RANGE_BORDER_PIX = 3

            calib_map_col = np.squeeze(orb.utils.image.nanbin_image(
                calib_map_col, binning))
            coeff_map_col = np.squeeze(orb.utils.image.nanbin_image(
                coeff_map_col, binning))
            mask_col = np.squeeze(orb.utils.image.nanbin_image(
                mask_col.astype(float), binning))
            mask_col[mask_col > 0] = 1
            data_col = np.empty((data.shape[1]/binning, data.shape[2]))
            for iz in range(data.shape[2]):
                # bin and convert mean binning to sum binning
                data_col[:,iz] = orb.utils.image.nanbin_image(
                    data[:,:,iz], binning) * binning**2
            data = data_col
            
            fit = np.empty((data.shape[0], len(lines), 6),
                           dtype=float)
            fit.fill(np.nan)
            
            err = np.empty((data.shape[0], len(lines), 6),
                           dtype=float)
            err.fill(np.nan)
                        
            for ij in range(data.shape[0]):
                if mask_col[ij]:

                    ## SUBTRACT SPECTRUM #############
                    if subtract is not None:
                        # convert lines wavelength/wavenumber into
                        # uncalibrated positions
                        if wavenumber:
                            axis = orb.utils.spectrum.create_cm1_axis(
                                data.shape[1], step, order,
                                corr=coeff_map_col[ij]).astype(float)
                        else:
                            axis = orb.utils.spectrum.create_nm_axis(
                                data.shape[1], step, order,
                                corr=coeff_map_col[ij]).astype(float)
                            
                        # interpolate and subtract                       
                        data[ij,:] -= subtract(axis)
                
                    ## FIT #############################
                        
                    # get signal range
                    min_range = np.nanmin(filter_range)
                    max_range = np.nanmax(filter_range)
                    
                    # adjust fwhm with incident angle (calib map must be real)
                    ifwhm_guess = fwhm_guess * calib_map_col[ij] / nm_laser

                    # check init velocity (already computed from
                    # binned maps). It a guess exists, velocity range
                    # is set to None
                    if not np.isnan(init_velocity_map_col[ij]):
                        ivelocity_range = None
                        shift_guess = np.array(shift_guess)
                        shift_guess.fill(init_velocity_map_col[ij])
                    else:
                        if velocity_range is not None:
                            ivelocity_range = float(velocity_range)
                        else:
                            ivelocity_range = None
                            
                    # check init sigma (already computed from
                    # binned maps). 
                    if not np.isnan(init_sigma_map_col[ij]):
                        sigma_guess = init_sigma_map_col[ij]
                    else:
                        sigma_guess = 0.
                        
                    try:
                        warnings.simplefilter('ignore')
                        result_fit = orb.fit.fit_lines_in_spectrum(
                            data[ij,:],
                            lines,
                            step, order,
                            nm_laser,
                            coeff_map_col[ij] * nm_laser,
                            fwhm_guess=ifwhm_guess,
                            shift_guess=shift_guess,
                            sigma_guess=sigma_guess,
                            cont_guess=None,
                            poly_order=poly_order,
                            cov_pos=cov_pos,
                            cov_sigma=cov_sigma,
                            cov_fwhm=cov_fwhm,
                            fix_fwhm=fix_fwhm, 
                            fmodel=fmodel,
                            signal_range=[min_range, max_range],
                            wavenumber=wavenumber,
                            apodization=apodization,
                            velocity_range=ivelocity_range)
                        warnings.simplefilter('default')
        
                    except Exception, e:
                        warnings.warn('Exception occured during fit: {}'.format(e))
                        import traceback
                        print traceback.format_exc()
                        
                        result_fit = []
                        
                else: result_fit = []

                for iline in range(len(lines)):
                    if result_fit != []:
                        ## print result_fit['velocity'], result_fit['velocity-err']
                        ## import pylab as pl
                        ## pl.plot(data[ij,:])
                        ## pl.plot(result_fit['fitted-vector'])
                        ## pl.show()
                        ## quit()
                    
                        fit[ij,iline,:5] = result_fit[
                            'lines-params'][iline, :]
                       
                        if 'lines-params-err' in result_fit:
                            err[ij,iline,:5] = result_fit[
                                'lines-params-err'][iline, :]
                        else:
                            err[ij,iline,:] = np.nan
                        
                        fit[ij,iline,2] = result_fit[
                            'velocity'][iline]
                        if 'velocity-err' in result_fit:
                            err[ij,iline,2] = result_fit[
                                'velocity-err'][iline]
                        else:
                            err[ij,iline,2] = np.nan

                        fit[ij,iline,4] = result_fit[
                            'broadening'][iline]
                        if 'broadening-err' in result_fit:
                            err[ij,iline,4] = result_fit[
                                'broadening-err'][iline]
                        else:
                            err[ij,iline,4] = np.nan

                        fit[ij,iline,5] = result_fit[
                            'flux'][iline]
                        if 'flux-err' in result_fit:
                            err[ij,iline,5] = result_fit[
                                'flux-err'][iline]
                        else:
                            err[ij,iline,5] = np.nan
                        
                            
                    else:
                        fit[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN')]
                        err[ij,iline,:] = [float('NaN'), float('NaN'),
                                           float('NaN'), float('NaN'),
                                           float('Nan'), float('NaN')]                

            return fit, err
        
        def _get_binned_dims(index, dimx, dimy, quad_nb, binning):
            div_nb = math.sqrt(quad_nb)
            quad_dimx = int(int(dimx / div_nb) / binning) * binning
            quad_dimy = int(int(dimy / div_nb) / binning) * binning
            index_x = index % div_nb
            index_y = (index - index_x) / div_nb
            x_min = long(index_x * quad_dimx)
            x_max = long((index_x + 1) * quad_dimx)
            y_min = long(index_y * quad_dimy)
            y_max = long((index_y + 1) * quad_dimy)
            return x_min, x_max, y_min, y_max        

        ## init LineMaps object    
        linemaps = LineMaps(self.dimx, self.dimy, lines, wavenumber,
                            binning, self.DIV_NB,
                            project_header=self._project_header,
                            wcs_header=self._wcs_header,
                            config_file_name=self.config_file_name,
                            data_prefix=self._data_prefix,
                            ncpus=self.ncpus)
        
        # compute max uncertainty on velocity and sigma to use it as
        # an initial guess.
        if wavenumber:
            resolution = (
                orb.cutils.get_cm1_axis_max(self.dimz, step, order)
                / fwhm_guess)
        else:
            resolution = (
                orb.cutils.get_nm_axis_max(self.dimz, step, order)
                / fwhm_guess)
            
        max_vel_err = (orb.constants.LIGHT_VEL_KMS / resolution) / 4.
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
        if np.all(subtract == 0.): subtract = None
                    
        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[region] = True

        for iquad in range(0, self.QUAD_NB):
            if binning > 1:
                x_min, x_max, y_min, y_max = _get_binned_dims(
                    iquad, self.dimx, self.dimy, self.QUAD_NB, binning)
            else:
                x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)

            # avoid loading quad with no pixel to fit in it
            if np.any(mask[x_min:x_max, y_min:y_max]):
                
                iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                           0, self.dimz)
                
                # multi-processing server init
                job_server, ncpus = self._init_pp_server()
                progress = orb.core.ProgressBar(x_max - x_min)
                for ii in range(0, x_max - x_min, ncpus*binning):
                    # no more jobs than columns
                    if (ii + ncpus*binning >= x_max - x_min): 
                        ncpus = (x_max - x_min - ii)/binning

                    # jobs creation
                    jobs = [(ijob, job_server.submit(
                        _fit_lines_in_column,
                        args=(iquad_data[ii+ijob*binning:ii+(ijob+1)*binning,:,:],
                              calibration_coeff_map[x_min + ii + ijob*binning:
                                                    x_min + ii + (ijob+1)*binning,
                                                    y_min:y_max],
                              
                              calibration_laser_map[x_min + ii + ijob*binning:
                                                    x_min + ii + (ijob+1)*binning,
                                                    y_min:y_max],
                              nm_laser,
                              mask[x_min + ii + ijob*binning:
                                   x_min + ii + (ijob+1)*binning,
                                   y_min:y_max],
                              lines, fwhm_guess, filter_range,
                              poly_order, fix_fwhm, fmodel,
                              step, order, wavenumber,
                              cov_pos, cov_fwhm, subtract, shift_guess, cov_sigma,
                              binning, apodization, velocity_range,
                              np.squeeze(
                                  init_velocity_map[
                                      (x_min + ii + ijob*binning)/binning:
                                      (x_min + ii + (ijob+1)*binning)/binning,
                                      y_min/binning:y_max/binning]),
                              np.squeeze(
                                  init_sigma_map[
                                      (x_min + ii + ijob*binning)/binning:
                                      (x_min + ii + (ijob+1)*binning)/binning,
                                      y_min/binning:y_max/binning])),
                        
                        modules=("import numpy as np",
                                 "import orb.fit",
                                 "import orb.utils.fft",
                                 "import orb.utils.spectrum",
                                 "import orcs.utils as utils",
                                 "import warnings", 'import math',
                                 'import inspect')))
                            for ijob in range(ncpus)]

                    for ijob, job in jobs:
                        (fit, err) = job()
                        x_range = ((x_min+ii+ijob*binning)/binning, (x_min+ii+(ijob+1)*binning)/binning)
                        y_range = (y_min/binning, y_max/binning)
                        linemaps.set_map('height', fit[:,:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude', fit[:,:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity', fit[:,:,2],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm', fit[:,:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('sigma', fit[:,:,4],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('flux', fit[:,:,5],
                                         x_range=x_range, y_range=y_range)
                        
                        linemaps.set_map('height-err', err[:,:,0],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('amplitude-err', err[:,:,1],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('velocity-err', err[:,:,2],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('fwhm-err', err[:,:,3],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('sigma-err', err[:,:,4],
                                         x_range=x_range, y_range=y_range)
                        linemaps.set_map('flux-err', err[:,:,5],
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
                                calibration_laser_map_path,
                                wavelength_calibration,
                                nm_laser,
                                poly_order=0,
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
        
        :param calibration_laser_map_path: Path to the calibration
          laser map.

        :param wavelength_calibration: True if the cube is calibrated.
        
        :param nm_laser: Calibration laser wavelentgh.

        :param poly_order: (Optional) Order of the polynomial used to fit
          continuum (default 0).

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
            calibration_laser_map_path, nm_laser, wavelength_calibration,
            axis_corr=axis_corr)
                
        self._print_msg("Extracting sky median vector")
        median_sky_spectrum = self._extract_spectrum_from_region(
            orb.utils.misc.get_mask_from_ds9_region_file(
                sky_regions_file_path,
                [0, self.dimx],
                [0, self.dimy]),
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

        lines, lines_name = orb.core.Lines().get_sky_lines(
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
            cov_sigma=True,
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
            
    def fit_lines_maps(self, object_regions_file_path, lines,
                           lines_velocity,
                           lines_fwhm, filter_range,
                           wavenumber, step, order,
                           calibration_laser_map_path,
                           wavelength_calibration,
                           nm_laser,
                           poly_order=0,
                           sky_regions_file_path=None,
                           cov_pos=True,
                           cov_sigma=True,
                           cov_fwhm=True,
                           fix_fwhm=False,
                           fmodel='gaussian',
                           axis_corr=1., binning=1,
                           apodization=None,
                           velocity_range=None):
        
        """
        Return emission lines parameters maps from a fit.

        All parameters of each emission line are mapped. So that for a
        gaussian fit with 4 parameters (height, amplitude, shift,
        fwhm) 4 maps are created for each fitted emission line in the
        cube.

        :param object_regions_file_path: Path to a ds9 regions file
          decribing the regions to fit.

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

        :param calibration_laser_map_path: Path tot the calibration laser map

        :param wavelength_calibration: True if the cube is calibrated.
        
        :param nm_laser: Calibration laser wavelentg in nm.

        :param poly_order: (Optional) Order of the polynomial used to fit
          continuum (default 0).

        :sky_regions_file_path: (Optional) Path to a ds9 region file
          giving the pixels where the sky spectrum has to be
          extracted. This spectrum will be subtracted to the spectral
          cube before it is fitted (default None).

        :param fix_fwhm: If True fix FWHM to its guess (default
          False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param cov_pos: (Optional) Lines positions in each spectrum
          are covarying.
    
        :param cov_sigma: (Optional) Lines sigma in each spectrum
          are covarying.

        :param cov_fwhm: (Optional) FWHM is the same for all the lines
          in each spectrum.

        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.

        :binning: (Optional) On the fly data binning (default 1).
        
        :param apodization: (Optional) Apodization level of the
          spectra. Line broadening due to apodization is
          removed from the fitting results (default None).

        :param velocity_range: (Optional) A float setting the range of
          velocities in km/s where the lines can be found (e.g. for a
          galaxy, default None)
        """
        self._print_msg("Extracting lines data", color=True)
        
        calibration_laser_map = self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser,
            False, # must be set to False to pass a real calibration
                   # laser map
            axis_corr=axis_corr)
        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser, wavelength_calibration,
            axis_corr=axis_corr)
            
        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            self._print_msg("Extracting sky median vector")
            sky_region = orb.utils.misc.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    [0, self.dimx],
                    [0, self.dimy])
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
        #if object_regions_file_path is None:
    
        region = orb.utils.misc.get_mask_from_ds9_region_file(
                    object_regions_file_path,
                    [0, self.dimx],
                    [0, self.dimy])
        linemaps = self._fit_lines_in_cube(
            region, lines, step, order, lines_fwhm,
            wavenumber, calibration_coeff_map,
            calibration_laser_map, nm_laser,
            poly_order, filter_range, lines_velocity,
            fix_fwhm=fix_fwhm,
            fmodel=fmodel,
            cov_pos=cov_pos,
            cov_sigma=cov_sigma,
            cov_fwhm=cov_fwhm,
            subtract=median_sky_spectrum,
            binning=binning,
            apodization=apodization,
            velocity_range=velocity_range)

        ## SAVE MAPS ###
        linemaps.write_maps()            
                    

    def fit_integrated_spectra(self, regions_file_path,
                               lines, lines_velocity,
                               lines_fwhm, filter_range,
                               wavenumber, step, order,
                               calibration_laser_map_path,
                               wavelength_calibration,
                               nm_laser,
                               poly_order=0,
                               sky_regions_file_path=None,
                               cov_pos=True,
                               cov_sigma=True,
                               cov_fwhm=True,
                               fix_fwhm=False,
                               fmodel='gaussian', 
                               plot=True,
                               auto_sky_extraction=False,
                               sky_size_coeff=2., axis_corr=1.,
                               verbose=True,
                               apodization=None,
                               velocity_range=None):
        """
        Fit integrated spectra and their emission lines parameters.

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
        
        :param calibration_laser_map_path: Path tot the calibration laser map

        :param wavelength_calibration: True if the cube is calibrated.
        
        :param nm_laser: Calibration laser wavelentg in nm.

        :param poly_order: Order of the polynomial used to fit
          continuum.
    
        :sky_regions_file_path: (Optional) Path to a ds9 region file
          giving the pixels where the sky spectrum has to be
          extracted. This spectrum will be subtracted to the spectral
          cube before it is fitted (default None).

        :param fix_fwhm: If True fix FWHM to its guess (default
          False).

        :param fmodel: (Optional) Type of line to fit. Can be
          'gaussian' or 'sinc' (default 'gaussian').

        :param cov_pos: (Optional) Lines positions in each spectrum
          are covarying.

        :param cov_sigma: (Optional) Lines sigma in each spectrum
          are covarying.


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

        :param verbose: (Optional) If True print the fit results
          (default True).

        :param apodization: (Optional) Apodization level of the
          spectra. Line broadening due to apodization is
          removed from the fitting results(default None).

        :param velocity_range: (Optional) A float setting the range of
          velocities in km/s where the lines can be found (e.g. for a
          galaxy, default None)    
        """
        def fit_region(region_index, lines, nm_laser, lines_velocity,
                       poly_order, cov_pos, cov_sigma, cov_fwhm,
                       fix_fwhm, fmodel, apodization, velocity_range,
                       spectrum, calibration_coeff_map,
                       calibration_laser_map, region, dimz, wavenumber,
                       step, order, auto_sky_extraction,
                       median_sky_spectrum, filter_range, lines_fwhm):
            warnings.simplefilter('ignore', RuntimeWarning)


            icorr = np.nanmean(calibration_coeff_map[region])
            icalib = np.nanmean(calibration_laser_map[region])
                        
            if wavenumber:
                axis = orb.utils.spectrum.create_cm1_axis(
                    dimz, step, order, corr=icorr)
            else:
                axis = orb.utils.spectrum.create_nm_axis(
                    dimz, step, order, corr=icorr)

            if auto_sky_extraction:
                # subtract sky around region
                raise Exception('Not implemented yet')              

            
            # interpolate
            spectrum = spectrum(axis.astype(float))

            if median_sky_spectrum is not None:
                spectrum -= median_sky_spectrum(axis.astype(float))

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
                    icorr * nm_laser,
                    fwhm_guess=ilines_fwhm,
                    shift_guess=lines_velocity,
                    cont_guess=None,
                    poly_order=poly_order,
                    cov_pos=cov_pos,
                    cov_sigma=cov_sigma,
                    cov_fwhm=cov_fwhm,
                    fix_fwhm=fix_fwhm,
                    fmodel=fmodel,
                    signal_range=[min_range, max_range],
                    wavenumber=wavenumber,
                    apodization=apodization,
                    velocity_range=velocity_range)
                
            except Exception, e:
                warnings.warn('Exception occured during fit: {}'.format(e))
                import traceback
                print traceback.format_exc()

                result_fit = []

            # return if bad fit
            if result_fit == []:
                return list(), spectrum, np.zeros_like(spectrum), axis

            fit_params = result_fit['lines-params']
            if 'lines-params-err' in result_fit:
                err_params = result_fit['lines-params-err']
                snr = result_fit['snr']

            else:
                err_params = np.empty_like(fit_params)
                err_params.fill(np.nan)
                snr = np.empty(fit_params.shape[0])
                snr.fill(np.nan)

            velocities = result_fit['velocity']
            if 'velocity-err' in result_fit:
                velocities_err = result_fit['velocity-err']
            else:
                velocities_err = np.empty_like(velocities)
                velocities_err.fill(np.nan)

            broadening = result_fit['broadening']
            if 'broadening-err' in result_fit:
                broadening_err = result_fit['broadening-err']
            else:
                broadening_err = np.empty_like(broadening)
                broadening_err.fill(np.nan)


            flux = result_fit['flux']
            if 'flux-err' in result_fit:
                flux_err = result_fit['flux-err']
            else:
                flux_err = np.empty_like(flux)
                flux_err.fill(np.nan)

            all_fit_results = list()
            for iline in range(fit_params.shape[0]):
                if wavenumber:
                    line_name = orb.core.Lines().round_nm2ang(
                        orb.utils.spectrum.cm12nm(
                            lines[iline]))
                else:
                    line_name = orb.core.Lines().round_nm2ang(
                        lines[iline])
                fit_results = {
                    'reg_index': region_index,
                    'line_name': line_name,
                    'h': fit_params[iline, 0],
                    'a': fit_params[iline, 1],
                    'x': fit_params[iline, 2],
                    'v': velocities[iline],
                    'fwhm': fit_params[iline, 3],
                    'sigma': fit_params[iline, 4],
                    'broadening': broadening[iline],
                    'flux': flux[iline],
                    'h_err': err_params[iline, 0],
                    'a_err': err_params[iline, 1],
                    'x_err': err_params[iline, 2],
                    'v_err': velocities_err[iline],
                    'broadening_err': broadening_err[iline],
                    'fwhm_err': err_params[iline, 3],
                    'sigma_err': err_params[iline, 4],
                    'snr': snr[iline],
                    'flux_err': flux_err[iline]}
                all_fit_results.append(fit_results)

            
            return all_fit_results, spectrum, result_fit['fitted-vector'], result_fit['fitted-models'], axis


        if verbose:
            self._print_msg("Extracting integrated spectra", color=True)
        
        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser,
            wavelength_calibration, axis_corr=axis_corr)

        calibration_laser_map = self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser, False,
            axis_corr=axis_corr)
                
        # Create parameters file
        paramsfile = orb.core.ParamsFile(
            self._get_integrated_spectra_fit_params_path())
        
        # Extract median sky spectrum
        if sky_regions_file_path is not None:
            if verbose:
                self._print_msg("Extracting sky median vector")
            median_sky_spectrum = self._extract_spectrum_from_region(
                orb.utils.misc.get_mask_from_ds9_region_file(
                    sky_regions_file_path,
                    [0, self.dimx],
                    [0, self.dimy]),
                calibration_coeff_map,
                wavenumber, step, order, median=True)
        else:
            median_sky_spectrum = None
            
        # extract regions
        integ_spectra = list()
        
        regions = orb.utils.misc.get_mask_from_ds9_region_file(
            regions_file_path,
            [0, self.dimx],
            [0, self.dimy],
            integrate=False)
        
        job_server, ncpus = self._init_pp_server()
        progress = orb.core.ProgressBar(len(regions))
        for iregion in range(0, len(regions), ncpus):
            progress.update(
                iregion,
                info="region %d/%d"%(iregion, len(regions)))
            if iregion + ncpus >= len(regions):
                ncpus = len(regions) - iregion

            jobs = [(ijob, job_server.submit(
                fit_region, 
                args=(iregion + ijob, lines, nm_laser, lines_velocity,
                      poly_order, cov_pos, cov_sigma, cov_fwhm,
                      fix_fwhm, fmodel, apodization, velocity_range,
                      self._extract_spectrum_from_region(
                          regions[iregion+ijob], calibration_coeff_map,
                          wavenumber, step, order, silent=True),
                      calibration_coeff_map,
                      calibration_laser_map, regions[iregion+ijob],
                      self.dimz, wavenumber, step, order,
                      auto_sky_extraction,
                      median_sky_spectrum, filter_range,
                      lines_fwhm),
                modules=("import numpy as np",
                         "import orb.utils.spectrum",
                         "import orb.utils.image",
                         "import orb.fit",
                         "import warnings",
                         "import orb.core")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                all_fit_results, spectrum, fitted_vector, fitted_models, axis = job()
                # write spectrum and fit
                region_index = iregion + ijob
                spectrum_header = (
                    self._get_integrated_spectrum_header(
                        region_index, axis, wavenumber))

                self.write_fits(
                    self._get_integrated_spectrum_path(
                        region_index),
                    spectrum, fits_header=spectrum_header,
                    overwrite=self.overwrite, silent=True)
                self.write_fits(
                    self._get_integrated_spectrum_fit_path(
                        region_index),
                    fitted_vector,
                    fits_header=spectrum_header,
                    overwrite=self.overwrite, silent=True)
                
                for fit_results in all_fit_results:
                    paramsfile.append(fit_results)
                    
                    if verbose:
                        self._print_msg(
                            'Line: {} ----'.format(
                                fit_results['line_name']))
                        for ikey in fit_results:
                            print '{}: {}'.format(
                                ikey, fit_results[ikey])

                if plot:
                    import pylab as pl
                    ax1 = pl.subplot(211)
                    ax1.plot(axis, spectrum, c= '0.3',
                             ls='--', lw=1.5,
                             label='orig spectrum')
                    
                    for imod in range(len(fitted_models[0])):
                        ax1.plot(axis, fitted_models[0][imod] + fitted_models[-2], c= '0.5',
                                 ls='-', lw=1.5)

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
        self._close_pp_server(job_server)
        progress.end()

                    
        return paramsfile
        

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

        :param calibration_laser_map_path: Path tot the calibration laser map

        :param wavelength_calibration: True if the cube is calibrated.
        
        :param nm_laser: Calibration laser wavelentg in nm.         
        
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
            calibration_laser_map_path, nm_laser, wavelength_calibration,
            axis_corr=axis_corr)

        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                       0, self.dimz)

            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = orb.core.ProgressBar(x_max - x_min)
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
                line_name = orb.core.Lines().round_nm2ang(
                    orb.utils.spectrum.cm12nm(rest_frame_lines[iline]))
                unit = 'cm-1'
            else:
                line_name = orb.core.Lines().round_nm2ang(rest_frame_lines[iline])
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
        




#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCubeTweaker(HDFCube):

    """ORCS spectral cube manipulation class.

    .. note:: parent class HDFCube is the ORCS implementation of
      HDFCube.
    """
    def _get_tweaked_cube_path(self):
        """Return path to a tweaked cube"""
        return self._data_path_hdr + "tweaked_cube.hdf5"

    def _get_detection_frame_path(self):
        """Return path to the detection frame"""
        return self._data_path_hdr + "detection_frame.fits"

    def _get_detection_pos_frame_path(self):
        """Return path to the detection position frame"""
        return self._data_path_hdr + "detection_pos_frame.fits"



    def subtract_spectrum(self, spectrum, axis, wavenumber, step, order,
                          calibration_laser_map_path, wavelength_calibration,
                          nm_laser, axis_corr=1):
        """Subtract a spectrum to the spectral cube

        :param spectrum: Spectrum to subtract
        
        :param axis: Wavelength/wavenumber axis of the spectrum to
          subtract (in nm or cm-1). The axis unit must be the same as
          the spectra cube unit.
        
        :param wavenumber: True if the spectral cube is in wavenumber

        :param step: Step size (in nm)

        :param order: Folding order

        :param calibration_laser_map_path: Path to the calibration
          laser map

        :param nm_laser: Calibration laser wavelength in nm
        
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """

        def subtract_in_vector(_vector, spectrum_spline, step, order,
                               calibration_coeff,
                               wavenumber):

            if wavenumber:
                axis = orb.utils.spectrum.create_cm1_axis(
                    _vector.shape[0], step, order,
                    corr=calibration_coeff).astype(float)
            else:
                axis = orb.utils.spectrum.create_nm_axis(
                    _vector.shape[0], step, order,
                    corr=calibration_coeff).astype(float)


            _to_sub = spectrum_spline(axis)

            return _vector - _to_sub
            

        if axis is None:
            self._print_warning('Spectrum axis guessed to be the same as the cube axis')
            if wavenumber:
                axis = orb.utils.spectrum.create_cm1_axis(
                    self.dimz, step, order, corr=axis_corr).astype(float)
            else:
                axis = orb.utils.spectrum.create_nm_axis(
                    self.dimz, step, order, corr=axis_corr).astype(float)
            
        spectrum_spline = scipy.interpolate.UnivariateSpline(
            axis[~np.isnan(spectrum)], spectrum[~np.isnan(spectrum)],
            s=0, k=1, ext=1)

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser,
            wavelength_calibration, axis_corr=axis_corr)

        self._parallel_process_in_quads(subtract_in_vector,
                                        (spectrum_spline, step, order,
                                         calibration_coeff_map,
                                         wavenumber),
                                        ('import numpy as np',
                                         'import orb.utils.spectrum'),
                                        self._get_tweaked_cube_path())


    def detect_sources(self, wavenumber, step, order,
                       calibration_laser_map_path, wavelength_calibration,
                       nm_laser, filter_range, axis_corr=1, fast=True):
        """Detect emission line sources in the spectral cube

        
        :param wavenumber: True if the spectral cube is in wavenumber

        :param step: Step size (in nm)

        :param order: Folding order

        :param calibration_laser_map_path: Path to the calibration
          laser map

        :param nm_laser: Calibration laser wavelength in nm

        :param filter_range: Range of detection
        
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.

        :param fast: (Optional) Fast detection algorithm (with FFT
          convolution). Borders of the frame are wrong but process is
          much faster (default True).

        """

        def filter_frame(_frame, fast):
            BOX_SIZE = 3 # must be odd
            BACK_COEFF = 9 # must be odd
            
            if not fast:
                _res = orb.cutils.filter_background(_frame, BOX_SIZE, BACK_COEFF)
            else:
                kernel_box = np.ones((BOX_SIZE, BOX_SIZE), dtype=float)
                kernel_box /= np.nansum(kernel_box)
                kernel_back = np.ones((BOX_SIZE*BACK_COEFF,
                                       BOX_SIZE*BACK_COEFF), dtype=float)
                kernel_back[kernel_back.shape[0]/2 - kernel_box.shape[0]/2:
                            kernel_back.shape[0]/2 + kernel_box.shape[0]/2 + 1,
                            kernel_back.shape[1]/2 - kernel_box.shape[1]/2:
                            kernel_back.shape[1]/2 + kernel_box.shape[1]/2 + 1] = np.nan
                kernel_back /= np.nansum(kernel_back)

                kernel_back[np.nonzero(np.isnan(kernel_back))] = 0.

                _box = scipy.signal.fftconvolve(_frame, kernel_box, mode='same')
                _back = scipy.signal.fftconvolve(_frame, kernel_back, mode='same')
                _res = _box - _back
            
            return _res

        Z_SIZE = 10
        FILTER_BORDER = 0.1

        if fast: self._print_msg('Source detection using fast algorithm')
        else: self._print_msg('Source detection using slow (but better) algorithm')


        # get filter range
        if wavenumber:
            cm1_axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order, corr=axis_corr)
            filter_range_pix = orb.utils.spectrum.cm12pix(
                cm1_axis, filter_range)

        else:
            nm_axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order, corr=axis_corr)
            filter_range_pix = orb.utils.spectrum.nm2pix(
                nm_axis, filter_range)

        fr = max(filter_range_pix) - min(filter_range_pix)
        fr_b = int(float(fr) * FILTER_BORDER)
        fr_min = min(filter_range_pix) + fr_b
        fr_max = max(filter_range_pix) - fr_b
        filter_range_pix = (int(fr_min), int(fr_max))

        if wavenumber: unit = 'cm-1'
        else: unit = 'nm'
        
        self._print_msg('Signal range: {} {}, {} pixels'.format(filter_range, unit, filter_range_pix))
        ## each pixel is replaced by the mean of the values in a small
        ## box around it minus the median value of the background
        ## taken in a larger box

        # Init multiprocessing server
        dat_cube = None
        det_frame = np.zeros((self.dimx, self.dimy), dtype=float)
        argdet_frame = np.copy(det_frame)
        argdet_frame.fill(np.nan)

        job_server, ncpus = self._init_pp_server()
        progress = orb.core.ProgressBar(
            max(filter_range_pix) - min(filter_range_pix))        
        for iframe in range(int(min(filter_range_pix)),
                            int(max(filter_range_pix)), ncpus):
            progress.update(
                iframe - min(filter_range_pix),
                info="working on frame %d/%d"%(
                    iframe - min(filter_range_pix) + 1,
                    max(filter_range_pix) - min(filter_range_pix)))
            if iframe + ncpus >= self.dimz:
                ncpus = self.dimz - iframe

            if dat_cube is not None:
                if last_frame < iframe + ncpus:
                    dat_cube = None

            if dat_cube is None:
                res_z_size = ncpus*Z_SIZE
                if iframe + res_z_size > self.dimz:
                    res_z_size = self.dimz - iframe
                first_frame = int(iframe)
                last_frame = iframe + res_z_size
                dat_cube = self.get_data(0, self.dimx,
                                         0, self.dimy,
                                         iframe, last_frame,
                                         silent=False)
                ## dat_cube = np.ones(
                ##     (self.dimx, self.dimy,
                ##      last_frame - iframe), dtype=float) + 10 * iframe
                ## dat_cube[200:205,200:205,10] += 3.
                
                res_cube = np.empty_like(dat_cube)
                res_cube.fill(np.nan)


            jobs = [(ijob, job_server.submit(
                filter_frame,
                args=(dat_cube[:,:,iframe - first_frame + ijob], fast),
                modules=("import orb.cutils",
                         "import numpy as np",
                         "import scipy.signal")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                # filtered data is written in place of non filtered data
                res_cube[:,:,iframe - first_frame + ijob] = job()
                
            max_frame = np.nanmax(res_cube, axis=2)
            argmax_frame = np.nanargmax(res_cube, axis=2) + first_frame
            new_det = np.nonzero(max_frame > det_frame)
            det_frame[new_det] = max_frame[new_det]
            argdet_frame[new_det] = argmax_frame[new_det]
                

        self._close_pp_server(job_server)
        progress.end()

        self.write_fits(self._get_detection_frame_path(),
                            det_frame, overwrite=True)
        self.write_fits(self._get_detection_pos_frame_path(),
                        argdet_frame, overwrite=True)
            
        
