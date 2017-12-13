import pandas as pd
import numpy as np
import os.path
import process
import h5py
import logging
import orb.core
import pylab as pl
from matplotlib import gridspec
import orb.core
import gvar

class Catalog(object):

    def __init__(self, catalog_path, cube_path=None, debug=False):

        self.debug = debug
        self.logger = orb.core.Logger(debug=self.debug)
        
        if not isinstance(catalog_path, str): raise TypeError('catalog_path must be a path')
        self.path = str(catalog_path)
        self.cube_path = None
        if cube_path is not None:
            if not isinstance(cube_path, str): raise TypeError('cube_path must be a path')
            if not os.path.exists(cube_path): raise IOError('Cube path does not exist')
            self.cube_path = str(cube_path)
                
            
        if os.path.exists(self.path):
            try:
                self.cat = pd.read_hdf(self.path, 'table')
            except KeyError:
                self.cat = None
                
            if self.cube_path is None:
                self.cube_path = self.get_attr('cube_path')
            else:
                logging.warning('Cube path changed')
                
        else:
            self.cat = None
            if self.cube_path is None: raise StandardError('If creating a new catalog a path to a Spectral Cube must be given')
            logging.info('Creating new catalog')

        # update cube path
        logging.info('Cube path : {}'.format(self.cube_path))
        self.cube = process.SpectralCube(self.cube_path)
        self.set_attr('cube_path', str(self.cube_path))

        # init catalog
        if self.cat is None:
            self.cat = pd.DataFrame()

        self._init_object_column('spectrum')
        self._init_object_column('axis')
        self._init_object_column('fitted_vector')
        
        
    def _init_object_column(self, key):
        nanarray = np.atleast_1d(np.empty(len(self.cat), dtype=float))
        nanarray.fill(np.nan)
        if key not in self.cat:
            self.cat[key] = pd.Series(nanarray.astype(object), dtype=object)
        else:
            self.cat[key] = self.cat[key].astype(object)
            
    def has_attr(self, attr):
        with h5py.File(self.path, 'r') as f:
            if attr in f.attrs: return True
            else: return False

    def get_attr(self, attr):
        with h5py.File(self.path, 'r') as f:
            if attr in f.attrs:
                return f.attrs[attr]
            else: return None

    def set_attr(self, attr, value):
        with h5py.File(self.path, 'a') as f:
            f.attrs[attr] = value
            
    def append(self, x, y, r=3, shape='circle'):
            
        if shape not in ['circle', 'square']: raise ValueError('shape must be circle or square')
        
        x = np.atleast_1d(np.squeeze(np.array(x, dtype=float)))
        y = np.atleast_1d(np.squeeze(np.array(y, dtype=float)))
        
        if x.shape != y.shape: raise TypeError('x and y must have the same shape')
        if x.ndim > 1: raise TypeError('x dimensions must be <= 1')

        
        for i in range(x.size):
            new = pd.Series({'x':x[0], 'y':y[0], 'r':r, 'shape':shape})
            self.cat = self.cat.append(new, ignore_index=True)

    def fit(self, index, lines=None, **kwargs):
        if self.is_empty(index):
            if lines is None: raise ValueError('this source has never been fitted, please set lines to something else than None')
            
        else:
            if lines is None:
                lines = self[index].lines

        old_kwargs = self[index].to_dict()
        for key in old_kwargs.keys():
            if 'input_' not in key:
                old_kwargs.pop(key)
                # old_kwargs.pop('x')
                # old_kwargs.pop('y')
                # old_kwargs.pop('r')
                # old_kwargs.pop('shape')
            else:
                old_kwargs[key[len('input_'):]] = old_kwargs[key]
                old_kwargs.pop(key)
        old_kwargs.update(kwargs)
        for ikey in self.cube.params:
            if ikey in old_kwargs:
                old_kwargs.pop(ikey)
        if 'pos_guess' in old_kwargs:
            old_kwargs.pop('pos_guess')
        if 'filter_function' in old_kwargs:
            old_kwargs.pop('filter_function')

        for ikey in old_kwargs.keys():
            print old_kwargs[ikey], type(old_kwargs[ikey])
            if type(old_kwargs[ikey]) not in (str, gvar.GVar):
                if isinstance(old_kwargs[ikey], np.ndarray):
                    if old_kwargs[ikey].dtype == object:
                        continue
                if np.any(np.isnan(old_kwargs[ikey])):
                    old_kwargs.pop(ikey)
            # try: ok = np.any(np.isnan(old_kwargs[ikey]))
            # except Exception: ok = True
            # if not ok: old_kwargs.pop(ikey)
    
        axis, spectrum, fit = self.cube.fit_lines_in_spectrum(self[index].x, self[index].y,
                                                              self[index].r, lines=lines,
                                                              **old_kwargs)
        
        ip = self.cube.inputparams.convert().allparams
        
        self._set_from_dict(index, ip, prefix='input_')

        self._init_object_column('spectrum')
        self._init_object_column('axis')
        
        self.cat.at[index, 'axis'] = [axis,]
        self.cat.at[index, 'spectrum'] = [spectrum]
        self._init_object_column('fitted_vector')
        self._set_from_dict(index, fit.convert())
        
    def _set_from_dict(self, index, ddict, prefix=''):
        for ikey in ddict.keys():
            if prefix + ikey not in self.cat:
                if type(ddict[ikey]) in [list, np.ndarray, tuple]:
                    self._init_object_column(prefix + ikey)
                if type(ddict[ikey]) in [dict]: # these types are not saved
                    continue
            self.cat.at[index, prefix + ikey] = ddict[ikey]

    def save(self):
        self.cat.to_hdf(self.path, 'table', mode='a')        
        
    def __getitem__(self, index):
        if self.cat is None:
            raise StandardError('Catalog is empty, please add at least one source with append(x, y)')
        
        if index in range(0, len(self.cat)):
            return self.cat.iloc[index]
        else: raise KeyError('Bad index. Catalog has {} elements'.format(len(self.cat)))

    
        
    def is_empty(self, index):
        if 'lines' in self[index]: return False
        return True

    def show(self, index=0):
        cv = CatalogViewer(self)
        cv.show(index)


class CatalogViewer(object):
    
    def __init__(self, cat):
        if not isinstance(cat, Catalog): raise TypeError('cat must be a orcs.catalog.Catalog instance')
        self.cat = cat
        
        self.fig = pl.figure(figsize=(10, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
                    
        # define grid
        gs  = gridspec.GridSpec(2, 3)
        self.ax0 = pl.subplot(gs[0,0]) # image0
        self.ax1 = pl.subplot(gs[0,1]) # image1
        self.ax2 = pl.subplot(gs[0,2]) # image2
        self.ax3 = pl.subplot(gs[1,:]) # spectrum

        self.axes = (self.ax0, self.ax1, self.ax2, self.ax3)

    def show(self, index):

        # get data
        axis = orb.core.Axis(np.array(self.cat[index].axis))
        spectrum = orb.core.Vector1d(np.array(self.cat[index].spectrum))
        fitted_vector = orb.core.Vector1d(np.array(self.cat[index].fitted_vector))

        # plot spectrum
        self.ax3.plot(axis.data, spectrum.data)
        self.ax3.plot(axis.data, fitted_vector.data)
        for iline in self.cat[index].input_pos_guess:
            _iline = gvar.mean(iline + orb.utils.spectrum.line_shift(
                iline, gvar.mean(self.cat[index].input_pos_cov), wavenumber=True))
            self.ax3.axvline(x=_iline)
        pl.show()
        
    def on_key(self, event):
        pass
        #if event.key == 'f':
            
