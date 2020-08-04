#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: process.py

## Copyright (c) 2010-2018 Thomas Martin <thomas.martin.1@ulaval.ca>
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

from . import version
__version__ = version.__version__

# import Python libraries
import os
import logging
import sys
import math
import time
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import bottleneck as bn
import warnings
import inspect
import scipy.interpolate
import marshal

# import core
from . import core
from . import fit
from .core import LineMaps
from . import utils

# import ORB
import orb.core
import orb.utils.spectrum
import orb.utils.image
import orb.utils.stats
import orb.utils.filters
import orb.utils.misc
import orb.utils.io
import orb.fit


#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCube(fit.SpectralCube):

    """ORCS spectral cube general processing class.

    .. note:: parent class SpectralCube is the ORCS implementation of
      SpectralCube.
    """

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

    def _get_deep_frame_wcs_path(self):
        """Return path to the generated deep frame with the reocmputed
        WCS.
        """
        return self._get_data_prefix() + 'wcs.deep_frame.fits'

    def _get_dxmap_path(self):
        """Return path to the generated X micro-shifting map."""
        return self._get_data_prefix() + 'wcs.dxmap.fits'

    def _get_dymap_path(self):
        """Return path to the generated Y micro-shifting map."""
        return self._get_data_prefix() + 'wcs.dymap.fits'


    def _get_calibration_laser_map_path(self):
        """Return path to the calibration map computed when fitting
        sky lines. Can be used instead of the original calibration
        map.
        """
        return self._get_data_prefix() + 'calibration_laser_map.fits'

    def _get_wavefront_map_path(self):
        """Return path to the wavefront map computed when fitting the
        calibration laser map with a different calibration laser
        wavelength. Can be used instead of the original wavefront map
        during a reduction
        """
        return self._get_data_prefix() + 'wavefront_map.fits'

    def _get_detection_frame_path(self):
        """Return path to the detection frame"""
        return self._data_path_hdr + "detection_frame.fits"

    def _get_detection_pos_frame_path(self):
        """Return path to the detection position frame"""
        return self._data_path_hdr + "detection_pos_frame.fits"

    def map_sky_velocity(self, mean_sky_vel, div_nb=20, plot=True,
                         x_range=None, y_range=None,
                         exclude_reg_file_path=None,
                         no_fit=False, threshold=None,
                         sky_lines=None):
        """Map the sky velocity on a rectangular grid and interpolate it
        to return a map of the velocity zero point that can be
        subtracted to the returned velocity map of the cube fit.

        :param mean_sky_vel: (Optional) Mean sky velocity (in km/s).

        :param div_nb: (Optional) Number of division on one axis of
          the rectangular grid. The total number of points is div_nb^2
          (default 15).

        :param plot: (Optional) If True, output plots are shown
          (default True).

        :param x_range: (Optional) Range of pixels along the X axis
          where the velocity is measured (default None).

        :param y_range: (Optional) Range of pixels along the Y axis
          where the velocity is measured (default None).

        :param exclude_reg_file_path: (Optional) Region to exclude
          from the computation (must be a ds9 reg file)

        :param no_fit: (Optional) Do not repeat the fitting
          process. Only recompute the velocity map model.

        :param threshold: (Optional) If not None, this threshold on
          the velocity uncertainty is used in place of an automatic
          threshold.
        """
        BINNING = 6

        pixel_size = self.config['PIX_SIZE_CAM1']

        if div_nb < 2:
            raise Exception('div_nb must be >= 2')
        MAX_R = 30 # hack should be 100

        if not no_fit:
            if os.path.exists(self._get_skymap_file_path()):
                logging.info('fitting process already done. Do you really want to redo it again ?')
                try:
                    if input('type [yes]: ') != 'yes': no_fit = True
                except Exception:
                    no_fit = True

        if no_fit:
            logging.warn('Fitting process not done again, only the final sky map is computed')

        dimx = self.dimx
        dimy = self.dimy
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

        if not no_fit:
            logging.info('X range: {} {}, Y range: {} {}'.format(xmin, xmax, ymin, ymax))

            r = min((xmax - xmin), (ymax - ymin)) / float(div_nb + 2) / 2.
            r = min(r, MAX_R)
            logging.info('Radius: {}'.format(r))

            exclude_mask = np.zeros((dimx, dimy), dtype=bool)
            if exclude_reg_file_path is not None:
                exclude_mask[orb.utils.misc.get_mask_from_ds9_region_file(
                    exclude_reg_file_path,
                    [0, dimx], [0, dimy])] = True


            with orb.utils.io.open_file(self._get_temp_reg_path(), 'w') as f:
                f.write('image\n')
                for ix in np.linspace(xmin, xmax, div_nb + 2)[1:-1]:
                    for iy in np.linspace(ymin, ymax, div_nb + 2)[1:-1]:
                        if not exclude_mask[int(ix), int(iy)]:
                            regions.append((ix, iy, r))
                            f.write('circle({:.5f},{:.5f},{:.5f})\n'.format(
                                ix, iy, r))

            logging.info('{} regions to fit'.format(len(regions)))

            if sky_lines is None:
                sky_lines = self.get_sky_lines()
            else:
                orb.utils.validate.is_iterable(sky_lines, object_name='sky_lines')

            logging.info('{} sky lines to fit'.format(len(sky_lines)))

            # fit sky spectra
            paramsfile = self.fit_integrated_spectra(
                self._get_temp_reg_path(), sky_lines, plot=False, verbose=False,
                fmodel='sinc', pos_def=['1'] * len(sky_lines),
                pos_cov=[mean_sky_vel], fwhm_def=['fixed'] * len(sky_lines))

            lines_nb = len(sky_lines)

            # write results
            with orb.utils.io.open_file(self._get_skymap_file_path(), 'w') as f:
                for ireg in range(len(regions)):
                    if 'v' in paramsfile[ireg*lines_nb]:
                        iv = paramsfile[ireg*lines_nb]['v']
                        iv_err = paramsfile[ireg*lines_nb]['v_err']
                    else:
                        iv = np.nan
                        iv_err = np.nan
                    f.write('{} {} {} {}\n'.format(
                        regions[ireg][0], regions[ireg][1], iv, iv_err))

        # fit map
        with orb.utils.io.open_file(self._get_skymap_file_path(), 'r') as f:
            sky_vel_map = list()
            sky_vel_map_err = list()
            x = list()
            y = list()
            for line in f:
                line = np.array(line.strip().split(), dtype=float)
                x.append(line[0])
                y.append(line[1])
                sky_vel_map.append(line[2])
                sky_vel_map_err.append(line[3])

        x = np.array(x)
        y = np.array(y)
        sky_vel_map = np.array(sky_vel_map)
        logging.debug('x: {}'.format(x))
        logging.debug('y: {}'.format(y))
        logging.debug('v: {}'.format(sky_vel_map))

        nans = np.isnan(sky_vel_map)
        sky_vel_map[nans] = 0.
        sky_vel_map_err = np.array(sky_vel_map_err)
        sky_vel_map_err[sky_vel_map_err == 0.] = np.nan


        # remove excluded regions (if already fitted, e.g. when the
        # process is done another time with a different exclusion mask)
        if exclude_reg_file_path is not None:
            exclude_mask = np.zeros((dimx, dimy), dtype=bool)
            exclude_mask[orb.utils.misc.get_mask_from_ds9_region_file(
                exclude_reg_file_path,
                [0, dimx], [0, dimy])] = True

            for i in range(len(x)):
                if exclude_mask[int(x[i]), int(y[i])]:
                    x[i] = np.nan
                    y[i] = np.nan
                    sky_vel_map[i] = np.nan
                    sky_vel_map_err[i] = np.nan

        # remove bad fits
        if threshold is None:
            threshold = np.nanmedian(sky_vel_map_err) + 1. * np.std(
                orb.utils.stats.sigmacut(sky_vel_map_err, sigma=2.))

        sky_vel_map_err[sky_vel_map_err > threshold] = np.nan

        # fit velocity error model
        (model_calib_map, wf,
         final_sky_vel_map, new_nm_laser) = utils.fit_velocity_error_model(
            x, y, sky_vel_map, sky_vel_map_err,
            self.params.nm_laser,
            self.get_calibration_laser_map_orig(),
            pixel_size,
            binning=BINNING)

        sky_vel_map = -sky_vel_map

        # write new calibration laser map
        orb.utils.io.write_fits(
            self._get_calibration_laser_map_path(),
            model_calib_map, overwrite=True,
            fits_header=[('CALIBNM', new_nm_laser, 'Calibration laser wl (nm)')])

        # write wavefront laser map
        orb.utils.io.write_fits(
            self._get_wavefront_map_path(),
            wf, overwrite=True,
            fits_header=[('CALIBNM', new_nm_laser, 'Calibration laser wl (nm)')])

        # write new velocity correction map
        orb.utils.io.write_fits(
            self._get_skymap_fits_path(),
            final_sky_vel_map, overwrite=True)

        if plot:
            import pylab as pl
            vmin = np.nanpercentile(final_sky_vel_map.flatten(), 3)
            vmax = np.nanpercentile(final_sky_vel_map.flatten(), 97)

            fig = pl.figure()
            pl.imshow(final_sky_vel_map.astype(float).T,
                      interpolation='none', vmin=vmin, vmax=vmax,
                      origin='lower-left', cmap='viridis')
            pl.colorbar()
            pl.title('Sky velocity map model (km/s)')
            fig.savefig(self._get_data_prefix() + 'sky_map_model_full.pdf')
            fig.savefig(self._get_data_prefix() + 'sky_map_model_full.svg')

            # get velocity on grid points
            final_sky_vel_map = [
                final_sky_vel_map[int(x[i]), int(y[i])] for i in range(len(x))]

            fig = pl.figure()
            pl.scatter(x, y, c=final_sky_vel_map, vmin=vmin, vmax=vmax, s=30,
                       cmap='viridis')
            pl.xlim((0, dimx))
            pl.ylim((0, dimy))
            pl.colorbar()
            pl.title('Sky velocity map model projected on grid points (km/s)')

            fig.savefig(self._get_data_prefix() + 'sky_map_model.pdf')
            fig.savefig(self._get_data_prefix() + 'sky_map_model.svg')


            fig = pl.figure()
            pl.scatter(x, y, c=sky_vel_map, vmin=vmin, vmax=vmax, s=30,
                       cmap='viridis')
            pl.xlim((0, dimx))
            pl.ylim((0, dimy))
            pl.colorbar()
            pl.title('Original measured sky velocity (km/s)')

            fig.savefig(self._get_data_prefix() + 'sky_map.pdf')
            fig.savefig(self._get_data_prefix() + 'sky_map.svg')

            fig = pl.figure()
            pl.scatter(x, y, c=sky_vel_map_err,
                       vmin=np.nanpercentile(sky_vel_map_err, 5),
                       vmax=np.nanpercentile(sky_vel_map_err, 95),
                       s=30,
                       cmap='viridis')
            pl.xlim((0, dimx))
            pl.ylim((0, dimy))
            pl.colorbar()
            pl.title('Original measured sky velocity uncertainty (km/s)')

            fig.savefig(self._get_data_prefix() + 'sky_map_err.pdf')
            fig.savefig(self._get_data_prefix() + 'sky_map_err.svg')


            fig = pl.figure()
            diff = final_sky_vel_map - sky_vel_map
            pl.scatter(x,y, c=diff,
                       vmin=np.nanpercentile(diff, 5),
                       vmax=np.nanpercentile(diff, 95),
                       s=30, cmap='viridis')
            pl.xlim((0, dimx))
            pl.ylim((0, dimy))
            pl.colorbar()
            pl.title('Fit residual (km/s)')

            fig.savefig(self._get_data_prefix() + 'residual.pdf')
            fig.savefig(self._get_data_prefix() + 'residual.svg')


            fig = pl.figure()
            pl.hist(diff[np.nonzero(~np.isnan(diff))], bins=30, range=(-10, 10))
            pl.title('Fit residual histogram (median: {}, std: {})'.format(
                np.nanmedian(diff), np.nanstd(diff)))
            fig.savefig(self._get_data_prefix() + 'residual_histogram.pdf')
            fig.savefig(self._get_data_prefix() + 'residual_histogram.svg')

            pl.show()

    def detect_sources(self, fast=True):
        """Detect emission line sources in the spectral cube

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
                kernel_back[kernel_back.shape[0]//2 - kernel_box.shape[0]//2:
                            kernel_back.shape[0]//2 + kernel_box.shape[0]//2 + 1,
                            kernel_back.shape[1]//2 - kernel_box.shape[1]//2:
                            kernel_back.shape[1]//2 + kernel_box.shape[1]//2 + 1] = np.nan
                kernel_back /= np.nansum(kernel_back)

                kernel_back[np.nonzero(np.isnan(kernel_back))] = 0.

                _box = scipy.signal.fftconvolve(_frame, kernel_box, mode='same')
                _back = scipy.signal.fftconvolve(_frame, kernel_back, mode='same')
                _res = _box - _back

            return _res

        Z_SIZE = 10
        FILTER_BORDER = 0.1

        if fast: logging.info('Source detection using fast algorithm')
        else: logging.info('Source detection using slow (but better) algorithm')

        # get filter range
        filter_range_pix = self.get_filter_range_pix()

        fr = max(filter_range_pix) - min(filter_range_pix)
        fr_b = int(float(fr) * FILTER_BORDER)
        fr_min = min(filter_range_pix) + fr_b
        fr_max = max(filter_range_pix) - fr_b
        filter_range_pix = (int(fr_min), int(fr_max))

        logging.info('Signal range: {} {}, {} pixels'.format(
            self.params.filter_range, self.unit, filter_range_pix))
        ## each pixel is replaced by the mean of the values in a small
        ## box around it minus the median value of the background
        ## taken in a larger box

        # Init multiprocessing server
        det_frame = np.zeros((self.dimx, self.dimy), dtype=float)
        argdet_frame = np.copy(det_frame)
        argdet_frame.fill(np.nan)

        dat_cube_slices = np.linspace(int(min(filter_range_pix)),
                                      int(max(filter_range_pix)),
                                      10).astype(int)
        
        for islice in range(len(dat_cube_slices) - 1):
            zmin = dat_cube_slices[islice]
            zmax = dat_cube_slices[islice+1]

            logging.info('Extracting frames: {} to {} ({}/{} frames)'.format(
                zmin, zmax,
                zmax - dat_cube_slices[0],
                max(filter_range_pix) - min(filter_range_pix)))
            
            dat_cube = self.get_data(0, self.dimx,
                                     0, self.dimy,
                                     zmin, zmax,
                                     silent=False)

            res_cube = np.empty_like(dat_cube)
            res_cube.fill(np.nan)

            job_server, ncpus = self._init_pp_server(timeout=1000)
            
            for iframe in range(0, res_cube.shape[2], ncpus):
                if iframe + ncpus >= res_cube.shape[2]:
                    ncpus = res_cube.shape[2] - iframe

                jobs = [(ijob, job_server.submit(
                    filter_frame,
                    args=(dat_cube[:,:,iframe + ijob], fast),
                    modules=("import logging",
                             "import orb.cutils",
                             "import numpy as np",
                             "import scipy.signal")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # filtered data is written in place of non filtered data
                    ifiltered_frame = job()
                    res_cube[:,:,iframe + ijob] = ifiltered_frame
                    
            self._close_pp_server(job_server)

            imax_frame = np.nanmax(res_cube, axis=2)
            iargmax_frame = np.nanargmax(res_cube, axis=2) + zmin
            new_det = np.nonzero(imax_frame > det_frame)
            det_frame[new_det] = imax_frame[new_det]
            argdet_frame[new_det] = iargmax_frame[new_det]


            
        orb.utils.io.write_fits(self._get_detection_frame_path(),
                                det_frame, overwrite=True)
        orb.utils.io.write_fits(self._get_detection_pos_frame_path(),
                                argdet_frame, overwrite=True)


    def register(self, distortion_map_path=None):
        """Make a new registration of the cube.

        :param distortion_map: A path to a FITS image containing an
          SIP distortion model. It can be a registered image of a
          calibration field containing a lot of stars and taken during
          the same run as the science cube.
        """
        raise NotImplementedError('outdated...')
        deep_frame = self.get_deep_frame()
        if deep_frame is None:
            raise Exception('No deep frame is attached to the cube. Please run the last step of the reduction process again.')

        sip = None
        compute_distortion = True
        if distortion_map_path is not None:
            dist_map_hdu = orb.utils.io.read_fits(distortion_map_path, return_hdu_only=True)
            hdr = dist_map_hdu.header
            sip = pywcs.WCS(hdr, naxis=2, relax=True)
            # distortion are already defined and must not be recomputed
            compute_distortion = False

        astro = Astrometry(
            deep_frame,
            target_radec=(self.params.target_ra,
                          self.params.target_dec),
            target_xy=(self.params.target_x,
                       self.params.target_y),
            wcs_rotation=self.params.wcs_rotation,
            sip=sip,
            instrument=self.instrument)

        wcs, dxmap, dymap = astro.register(
            compute_distortion = compute_distortion,
            return_error_maps=True)

        newhdr = wcs.to_header(relax=True)
        newhdr['CD1_1'] = wcs.wcs.cd[0,0]
        newhdr['CD1_2'] = wcs.wcs.cd[0,1]
        newhdr['CD2_1'] = wcs.wcs.cd[1,0]
        newhdr['CD2_2'] = wcs.wcs.cd[1,1]

        orb.utils.io.write_fits(self._get_deep_frame_wcs_path(), deep_frame,
                        fits_header=newhdr, overwrite=True)
        orb.utils.io.write_fits(self._get_dxmap_path(), dxmap,
                        fits_header=newhdr, overwrite=True)
        orb.utils.io.write_fits(self._get_dymap_path(), dymap,
                        fits_header=newhdr, overwrite=True)


