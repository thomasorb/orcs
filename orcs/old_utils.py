#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orcs.py

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
Utils module contains core functions that are used by the processing
classes of ORCS
"""

def add_phase(a, phi):
    """Add a phase to a vector.

    :param a: A vector

    :param phi: Phase to be added. Can be a float or a vector of the
        same size as a.

    :return: a complex vector of the same size as a.
    """
    a = np.copy(a)
    a = a.astype(complex)
    a_mean = np.mean(a)
    a -= a_mean
    a_real = np.copy(a.real)
    a_imag = np.copy(a.imag)
    a.real = a_real * np.cos(phi) - a_imag * np.sin(phi)
    a.imag = a_imag * np.cos(phi) + a_real * np.sin(phi)
    a += a_mean
    return a



def read_sexphot(sexphot_path):
    """Read a SExtractor sources list file given with an HST frame
    
    :param sexphot_path: Path to the SExtractor source list
    """
    f = open(sexphot_path, 'r')
    all_phot = list()
    par = list()
    for iline in f:
        if 'X-Center' in iline:
            par = iline.split()[1:]
        if '#' not in iline:
            iline = iline.split()
            phot = dict()
            for ipar in range(len(par)):
                phot[par[ipar]]=float(iline[ipar])
            all_phot.append(phot)

    f.close()
    return all_phot

def sexcalib_hst_image(hst_path, sexphot_path, lambda_c):
    """Calibrate an HST frame using a Sextractor sources list.

    .. note:: WFPC2 frames can be calibrated by simply multiplying
        their count rate by the value of PHOTFLAM in the header of the
        frame.
    """
    sexphot = read_sexphot(sexphot_path)
    calib_coeff_list = list()
    for iphot in range(len(sexphot)):
        calib_coeff_list.append(
            orb.utils.ABmag2flambda(sexphot[iphot]['MagBest'], lambda_c)
            / sexphot[iphot]['FluxBest'])
    calib_coeff = np.mean(calib_coeff_list)
    print 'calibration coeff: %e [std: %e]'%(
        calib_coeff, np.std(calib_coeff_list))

    hdulist = pyfits.open(hst_path)
    hdulist[1].data *= calib_coeff
    hdulist.writeto(hst_path + '.sexcalib', clobber=True)

def calib_hst_image(hst_path):
    """Calibrate an HST frame using PHOTFLAM keyword."""

    hdulist = pyfits.open(hst_path)
    calib_coeff = hdulist[1].header['PHOTFLAM']
    
    print 'PHOTFLAM calibration coeff: %e'%calib_coeff
    
    hdulist[1].data *= calib_coeff
    hdulist.writeto(hst_path + '.calib', clobber=True)

    return hdulist[1].data.T


def simulate_hst_filter(cube_path, filter_file_path):
    """Simulate an HST image from a calibrated spectrum cube

    :param cube_path: Path to the calibrated spectrum cube.

    :param filter_file_path: Path to the HST filter file.
    """
    f = open(filter_file_path, 'r')
    fang = list()
    ftrans = list()
    for line in f :
        line = line.split()
        fang.append(float(line[0]))
        ftrans.append(float(line[1]))

    fang = np.array(fang)
    ftrans = np.array(ftrans)

    hdu = Tools().read_fits(cube_path, return_hdu_only=True)
    hdr = hdu[0].header
    data = hdu[0].data.T
    ang_axis = orb.utils.create_nm_axis(
        hdr['NAXIS3'], hdr['STEP'], hdr['ORDER']) * 10.

    ftrans = orb.utils.interpolate_axis(ftrans, ang_axis, 1, old_axis=fang)

    simu = (np.nansum((data * ftrans), axis=2)
            / (np.nansum(ftrans)))

    Tools().write_fits('simulated_frame.fits', simu, fits_header=hdr,
                       overwrite=True)

    return simu


def get_calibration_coeff_from_hst_image(cube_path, hst_path,
                                         filter_file_path,
                                         hst_ds9_reg_path,
                                         cube_ds9_reg_path,
                                         hst_sky_ds9_reg_path,
                                         sky_ds9_reg_path):
    """
    2 reg files are nescessary to be able to compare the same regions
    in the HST image and the simulated image from the cube. Those
    files must gives the coordinates of the same region in 'image'
    format (region can be a box or a circle).
    """

    hst_pix = orb.utils.get_mask_from_ds9_region_file(hst_ds9_reg_path)
    cube_pix = orb.utils.get_mask_from_ds9_region_file(cube_ds9_reg_path)
    sky_pix =  orb.utils.get_mask_from_ds9_region_file(sky_ds9_reg_path)
    hst_sky_pix =  orb.utils.get_mask_from_ds9_region_file(hst_sky_ds9_reg_path)
    
    hdu_cube = Tools().read_fits(cube_path, return_hdu_only=True)
    hdr_cube = hdu_cube[0].header
    hdu_hst = Tools().read_fits(hst_path, return_hdu_only=True)
    hdr_hst = hdu_hst[1].header
    
    cube_pix_area = abs(hdr_cube['CDELT1'] * hdr_cube['CDELT2']) * 3600**2.
    hst_pix_area = abs(hdr_hst['CD1_1'] * hdr_hst['CD2_2'] * 3600**2.)
    cube_area = len(cube_pix[0]) * cube_pix_area
    hst_area = len(hst_pix[0]) * hst_pix_area

    hst_image = calib_hst_image(hst_path).astype(float)
    
    
    sim_hst_image = simulate_hst_filter(cube_path, filter_file_path).astype(float)

    # get sky level in simulated frame
    #sky_level = orb.astrometry.sky_background_level(sim_hst_image)
    #sky_level = 0
    # get sky level in simulated frame
    sky_level = orb.utils.robust_median(sim_hst_image[sky_pix])
    sky_level_std = orb.utils.robust_std(sim_hst_image[sky_pix])
    hst_sky_level = orb.utils.robust_median(hst_image[sky_pix])
    hst_sky_level_std = orb.utils.robust_std(hst_image[sky_pix])
    
    hst_sum = orb.utils.robust_sum(hst_image[hst_pix] - hst_sky_level) / hst_area
    sim_hst_sum = orb.utils.robust_sum(
        sim_hst_image[cube_pix] - sky_level) / cube_area
    
    sky_level_error = np.sqrt((len(cube_pix[0]) * sky_level_std**2.)) / cube_area
    hst_sky_level_error = np.sqrt((len(hst_pix[0]) * hst_sky_level_std**2.)) / hst_area
    total_error = math.sqrt((sky_level_error/sim_hst_sum)**2. + (hst_sky_level_error/hst_sum)**2.)
    print 'sky_level: {}, error: {}%'.format(sky_level,
                                             sky_level_std/sky_level*100)
    print 'hst [flux mean / arcsec^2]: ', hst_sum
    print 'sim_hst [flux mean / arcsec^2]: ', sim_hst_sum
    print 'flux error: {}%'.format((total_error)*100.)
    calib_coeff = hst_sum / sim_hst_sum
    print calib_coeff
    return calib_coeff
    
  
