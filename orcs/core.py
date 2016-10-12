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
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import scipy.interpolate

# import ORB
try:
    import orb.core
    

except IOError, e:
    print "ORB could not be found !"
    print e
    sys.exit(2)


        

#################################################
#### CLASS Tools ################################
#################################################
class Tools(orb.core.Tools):
    """Extension of :py:class:`orb.core.Tools`

    .. seealso:: :py:class:`orb.core.Tools`
    """

    def _get_orcs_data_file_path(self, file_name):
        """Return the path to a file in ORCS data folder: orcs/data/file_name

        :param file_name: Name of the file in ORCS data folder.
        """
        return os.path.join(os.path.split(__file__)[0], "data", file_name)

#################################################
#### CLASS HDFCube ##############################
#################################################

class HDFCube(orb.core.HDFCube):
    """Extension of :py:class:`orb.core.HDFCube`

    .. seealso:: :py:class:`orb.core.HDFCube`
    """

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
          the columns). Must be f(vector_data, *args) or
          f(column_data, * args).

    
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
            self.QUAD_NB,
            reset=True)


        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, y_min, y_max, 
                                       0, self.dimz)
            ## iquad_data = np.zeros((x_max - x_min, y_max - y_min, self.dimz), dtype=float)
        

            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = orb.core.ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus * binning):
                progress.update(ii, 'processing quad{}/{}'.format(
                    iquad+1, self.QUAD_NB))
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
            self._print_msg('Writing quad {}/{} to disk'.format(
                iquad+1, self.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(iquad, data=iquad_data)
            self._print_msg('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.QUAD_NB, time.time() - write_start_time))

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
                                   nm_laser, wavelength_calibration,
                                   axis_corr=1):
        """Return the calibration laser map.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.

        :param nm_laser: Laser wavelength in nm
        
        :param wavelength_calibration: True if the cube is calibrated.
    
        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """
        if wavelength_calibration:
            return (np.ones((self.dimx, self.dimy), dtype=float)
                    * nm_laser * axis_corr)
        
        else:
            calibration_laser_map = self.read_fits(
                calibration_laser_map_path)
            if (calibration_laser_map.shape[0] != self.dimx):
                calibration_laser_map = orb.utils.image.interpolate_map(
                    calibration_laser_map, self.dimx, self.dimy)
            return calibration_laser_map

    def _get_calibration_coeff_map(self, calibration_laser_map_path,
                                   nm_laser, wavelength_calibration,
                                   axis_corr=1):
        """Return the calibration coeff map based on the calibration
        laser map and the laser wavelength.

        :param calibration_laser_map_path: Path to the calibration
          laser map. If None, the returned calibration laser map will
          be a map full of ones.

        :param nm_laser: calibration laser wavelength in nm.

        :param wavelength_calibration: True if the cube is calibrated

        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.

        """
        return (self._get_calibration_laser_map(
            calibration_laser_map_path, nm_laser, wavelength_calibration,
            axis_corr=axis_corr)
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
            base_axis[~np.isnan(spectrum)], spectrum[~np.isnan(spectrum)],
            s=0, k=1, ext=1)
         
        return spectrum_function


    def extract_integrated_spectrum(self, regions_file_path,
                                    wavenumber, step, order,
                                    calibration_laser_map_path,
                                    wavelength_calibration,
                                    nm_laser, axis_corr=1.):

        """Extract integrated spectrum over a given region
        
        :param regions_file_path: Path to a ds9 reg file giving the
          positions of the regions. Each region is considered as a
          different region.

        :param wavenumber: If True the cube is in wavenumber else it
          is in wavelength.

        :param step: Step size in nm

        :param order: Folding order
        
        :param calibration_laser_map_path: Path tot the calibration laser map

        :param wavelength_calibration: True if the cube is calibrated.
        
        :param nm_laser: Calibration laser wavelentg in nm.

        :param axis_corr: (Optional) If the spectrum is calibrated in
          wavelength but not projected on the interferometer axis
          (angle 0) the axis correction coefficient must be given.
        """

        calibration_coeff_map = self._get_calibration_coeff_map(
            calibration_laser_map_path, nm_laser,
            wavelength_calibration, axis_corr=axis_corr)

        if wavenumber:
            axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order, corr=axis_corr).astype(float)
        else:
            axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order, corr=axis_corr).astype(float)

        median_spectrum_spline = self._extract_spectrum_from_region(
            orb.utils.misc.get_mask_from_ds9_region_file(
                regions_file_path,
                [0, self.dimx],
                [0, self.dimy]),
            calibration_coeff_map,
            wavenumber, step, order, median=True)

        median_spectrum = median_spectrum_spline(axis)
        
        spectrum_header = (
            self._get_integrated_spectrum_header(
                0, axis, wavenumber))
        
        self.write_fits(
            self._get_integrated_spectrum_path(0),
            median_spectrum, fits_header=spectrum_header,
            overwrite=self.overwrite)

        return median_spectrum
        

#################################################
#### CLASS OrcsBase #############################
#################################################

class OrcsBase(Tools):
    """Load HDFCube supplementary data (header, calibration laser map)"""


    options = dict()
    config = dict()
    
    def __init__(self, spectrum_cube_path, **kwargs):
        """Init HDFCubeLoader class.

        :param spectrum_cube_path: Path to the spectral cube. 
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
        """                
                
        Tools.__init__(self, **kwargs)
        self.__version__ = version.__version__
        self.overwrite = True
        
        ## get config parameters
        self._store_config_parameter('OBS_LAT', float)
        self._store_config_parameter('OBS_LON', float)
        self._store_config_parameter('OBS_ALT', float)
        self._store_config_parameter("CALIB_NM_LASER", float)        

        # load cube
        cube = HDFCube(
            spectrum_cube_path,
            ncpus=self.ncpus,
            config_file_name=self.config_file_name)
        self.header = cube.get_cube_header()

        self.options['spectrum_cube_path'] = spectrum_cube_path
        
        # Observation parameters
        self._store_option_parameter('object_name', 'OBJECT', str)
        self._store_option_parameter('filter_name', 'FILTER', str)
        self._store_option_parameter('apodization', 'APODIZ', float)
        self._print_msg('Apodization: {}'.format(self.options['apodization']))
       
        self.options['project_name'] = (
            self.options['object_name']
            + '_' + self.options['filter_name']
            + '_' + str(self.options['apodization'])
            + '.ORCS')
        
        self._store_option_parameter('step', 'STEP', float)
        self._store_option_parameter('order', 'ORDER', int)
        self._store_option_parameter('axis_corr', 'AXISCORR', float)

        # wavenumber
        self._store_option_parameter('wavetype', 'WAVTYPE', str)
        if self.options['wavetype'] == 'WAVELENGTH':
            self.options['wavenumber'] = False
            self._print_msg('Cube is in WAVELENGTH (nm)')
            self.unit = 'nm'
        else:
            self.options['wavenumber'] = True
            self._print_msg('Cube is in WAVENUMBER (cm-1)')
            self.unit = 'cm-1'

        # wavelength calibration
        self._store_option_parameter(
            'wavelength_calibration', 'WAVCALIB', bool,
            optional=True)

        # get internal calibration map
        calib_map = cube.get_calibration_laser_map()       
        if calib_map is not None:
            calib_map_path = self._get_calibration_laser_map_path()
            self.write_fits(
                calib_map_path, calib_map, overwrite=True)
            self.options[
                'calibration_laser_map_path'] = calib_map_path
        else:
            self._store_option_parameter(
                'calibration_laser_map_path', 'CALIBMAP', str)
        
        self._print_msg('Calibration laser map used: {}'.format(
            self.options['calibration_laser_map_path']))
            
        if self.options['wavelength_calibration']:
            self._print_msg('Cube is CALIBRATED')
        else:
            self._print_msg('Cube is NOT CALIBRATED')
            
        ## Get WCS header        
        self.wcs = pywcs.WCS(self.header, naxis=2)
        self.wcs_header = self.wcs.to_header()

        # get ZPD index
        self.options['step_nb'] = cube.get_cube_header()['STEPNB']
        if 'ZPDINDEX' in cube.get_cube_header():
            self.zpd_index = cube.get_cube_header()['ZPDINDEX']
        else:
            self._print_error('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')

        
        self._store_option_parameter('target_ra', 'TARGETR', str, split=':',
                                     post_cast=float)
        self._store_option_parameter('target_dec', 'TARGETD', str, split=':',
                                     post_cast=float)
        self._store_option_parameter('target_x', 'TARGETX', float)
        self._store_option_parameter('target_y', 'TARGETY', float)
        self._store_option_parameter('obs_date', 'DATE-OBS', str, split='-',
                                     post_cast=int)
        self._store_option_parameter('hour_ut', 'HOUR_UT', str, split=':',
                                     post_cast=float, optional=True)
        if 'hour_ut' not in self.options:
            self.options['hour_ut'] = (0.,0.,0.)


    def _store_config_parameter(self, key, cast):
        """Store a configuration parameter

        :param key: Configuration file key

        :param cast: Type cast
        """
        if cast is not bool:
            self.config[key] = cast(
                self._get_config_parameter(key))
        else:
            self.config[key] = bool(int(
                self._get_config_parameter(key)))


    def _store_option_parameter(self, option_key, key, cast, optional=False,
                                split=None, post_cast=str):
        """Store an optional parameter

        :param option_key: Option Key

        :param key: Option file key

        :param cast: Type cast

        :param optional: (Optional) Key is optional (no error raised
          if key does not exist) (default None)

        :param split: (Optional) split key value after first cast (default None)

        :param post_cast: (Optional) Final cast on the key after first
          cast and split. Used only if split is nt None (default str).
        """

        if key in self.header:
            value = cast(self.header[key])
        else: value = None

        if value is not None:
            if split is not None:
                value = value.split(split)
                value = np.array(value).astype(post_cast)
            self.options[option_key] = value
        elif not optional: self._print_error(
            "Keyword '{}' must be set in the cube header".format(key))

    def _get_calibration_laser_map_path(self):
        """Return path to calibration laser map
        """
        return self._get_data_prefix() + 'calibration_laser_map.fits'

    def _get_data_prefix(self):
        """Return prefix path to stored data"""
        return (self._get_project_dir()
                + self.options['object_name']
                + '_' + self.options['filter_name']
                + '_' + str(self.options['apodization']) + '.')

    def _get_project_dir(self):
        """Return the path to the project directory depending on 
        the project name."""
        return os.curdir + os.sep + self.options["project_name"] + os.sep

    def _get_project_fits_header(self):
        """Return the header of the project that can be added to the
        created FITS files."""
        hdr = list()
        hdr.append(('ORCS', '{:s}'.format(self.__version__), 'ORCS version'))
        return hdr



##################################################
#### CLASS LineMaps ##############################
##################################################

class LineMaps(Tools):
    """Manage line parameters maps"""


    params = ('height', 'height-err', 'amplitude', 'amplitude-err',
              'velocity', 'velocity-err', 'fwhm', 'fwhm-err',
              'sigma', 'sigma-err', 'flux', 'flux-err')

    _wcs_header = None

    def __init__(self, dimx, dimy, lines, wavenumber, binning, div_nb,
                 project_header=None, wcs_header=None, **kwargs):
        """Init class
    
        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
        """
        Tools.__init__(self, **kwargs)
        self._project_header = project_header
        self._wcs_header = wcs_header
        self.__version__ = version.__version__

        self.wavenumber = wavenumber
        self.DIV_NB = div_nb
        self.binning = binning
        
        if binning > 1:
            self.dimx = int(int(dimx / self.DIV_NB) / self.binning) * self.DIV_NB
            self.dimy = int(int(dimy / self.DIV_NB) / self.binning) * self.DIV_NB
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

        self.data = dict()
        base_array =  np.empty((self.dimx, self.dimy, len(lines)),
                               dtype=float)
        base_array.fill(np.nan)
        for iparam in self.params:
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

        if param not in self.params:
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
                for param in self.params:
                    if not os.path.exists(self._get_map_path(
                        line_name, param, binning)):
                        all_ok = False
            if all_ok: available_binnings.append(binning)

        if len(available_binnings) < 1: return
        # load data from lowest (but still higher than requested)
        # binning
        binning = np.nanmin(available_binnings)
        self._print_msg('Loading {}x{} maps'.format(
            binning, binning))
        for param in self.params:
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
                self._print_msg('{} loaded'.format(param))
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

        if param not in self.params:
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




#################################################
#### CLASS SDSSFilter ###########################
#################################################

class SDSSFilter(Tools):

    filter_names = ['u', 'g', 'r', 'i', 'z']

    def __init__(self, filter_name):
        """Init class.

        :param filter_name: Name of the filter
        """
        if filter_name in self.filter_names:
            self.filter_name = filter_name
        else: self._print_error('Bad filter name: {}. Must be in {}'.format(
            filter_name, self.filter_names))

        self.filter_path = get_path()
        
    def get_path(self, filter_name=None):
        """Return filter file path

        :param filter_name: (Optional) Name of the filter. If None
          return the default filter path loaded during init (default
          None).
        """

        if filter_name is None:
            filter_name = self.filter_name

        _path = self._get_orcs_data_file_path(
            'filter_SDSS_' + filter_name + '.orc')

        if os.path.exists(_path): return _path
        else: self._print_error('Filter file: {} does not exist'.format(_path))


    def get_transmission(self, step, order, step_nb,
                         wavenumber=True, corr=1.):
        """Return the transmission function of the filter
        
        :param step: Step size of the moving mirror in nm.

        :param order: Folding order.

        :param step_nb: Number of points.

        :param wavenumber: (Optional) If True the function is
          interpolated and returned along a wavenumber axis. If False
          it is returned along a wavelength axis (default True).
    
        :param corr: (Optional) Correction coefficient related to the
          incident angle (default 1).
        """

        if not wavenumber:
            _axis = orb.utils.spectrum.create_nm_axis(
                n, step, order, corr=corr)
        else:
            _axis = orb.utils.spectrum.create_nm_axis_ireg(
                n, step, order, corr=corr)

        with self.open_file(self.filter_path, 'r') as f:
            for line in f:
                print line
            quit()


