#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: image.py

# Copyright (c) 2010-2018 Thomas Martin <thomas.martin.1@ulaval.ca>
#
# This file is part of ORCS
#
# ORCS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ORCS is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ORCS.  If not, see <http://www.gnu.org/licenses/>.

"""
ORCS Image library.

.. note:: ORCS is built over ORB so that ORB must be installed.
"""

import numpy as np
import orb.utils.validate
import warnings
import logging
import numbers

#################################################
#### CLASS Pixel ################################
#################################################


class Pixel(object):

    def __init__(self, data, ctype=int):

        
        if isinstance(data, Pixel):
            data = np.array(data.data)
        else:
            data = np.array(data, dtype=float)

        data[0] = ctype(data[0])
        data[1] = ctype(data[1])
        self.data = data
        self.x = ctype(data[0])
        self.y = ctype(data[1])
        self.xy = (self.x, self.y)
        
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return 'Pixel' + str(self.data)

    def __hash__(self):
        return hash(tuple(self.data))

    def __eq__(self, pix):
        if not isinstance(pix, Pixel):
            raise TypeError('pix must be a Pixel instance')
        # if np.any(self.data != pix.data) and np.all(self.xy == pix.xy):
        #     warnings.warn('pixels have same coordinates but different values')
        return np.any(self.data == pix.data)

    def astype(self, ctype):
        return Pixel(self, ctype=ctype)

    def validate(self, xmin, xmax, ymin, ymax):
        try:
            orb.utils.validate.index(self.x, xmin, xmax, clip=False)
            orb.utils.validate.index(self.y, ymin, ymax, clip=False)
        except orb.utils.err.ValidationError:
            return False
        return True

    def get_neighbors(self):
        if not hasattr(self, '_nngrid'):
            X,Y = np.mgrid[-1:2,-1:2]
            self._nngrid = np.array((X.flatten(), Y.flatten()))
        nns = set()
        for inn in (np.array((self.x,self.y)).reshape((2,1)) + self._nngrid).T:
            nns.add(tuple(inn))
        nns.remove((self.x, self.y))
        return nns

    def distance(self, pix):
        pix = Pixel(pix)
        return np.sqrt((pix.x - self.x)**2. + (pix.y - self.y)**2.)

    def get_pixels_in_radius(self, radius):
        def get_attr_name(radius):
            return '_r{:03d}_grid'.format(int(radius)+1)

        if not hasattr(self, get_attr_name(radius)):
            maxr = int(radius) + 1
            X,Y = np.mgrid[-maxr:maxr+1,-maxr:maxr+1]
            R = np.sqrt(X**2+Y**2)
            setattr(self, get_attr_name(radius),
                    (np.array((X.flatten(), Y.flatten())), R.flatten()))
            
        nns = set()
        XY, R = getattr(self, get_attr_name(radius))
        XY = (np.array((self.x,self.y)).reshape((2,1)) + XY).T
        for inn in range(XY.shape[0]):
            if R[inn] <= radius:
                nns.add(tuple(XY[inn,:]))
        return nns
    
    def get_neighbors_in_radius(self, radius):
        nns = self.get_pixels_in_radius(radius)
        nns.remove((self.x, self.y))
        return nns

        

        
#################################################
#### CLASS Source ###############################
#################################################
class Source(object):
    
    def __init__(self, data=None, frame_shape=None, ctype=int):

        if frame_shape is not None:
            orb.utils.validate.has_len(frame_shape, 2, object_name='frame_shape')
            self.maxx = int(frame_shape[0])
            self.maxy = int(frame_shape[1])
            self.shape = (self.maxx, self.maxy)
            self.check_pixels = True
        else:
            self.shape = None
            self.check_pixels = False

        self.pixels = set()
        self.pixel_size = None

        if data is not None:
            try:
                self.add(data, ctype=ctype)
            except TypeError, ValueError:                
                try:
                    self.update(data, ctype=ctype)
                except TypeError, ValueError:
                    raise TypeError('invalid data. Must be a Pixel instance, a tuple, a Source instance or a set of Pixels/tuples')
            
    def __hash__(self):
        return hash(self.pixels)

    def __eq__(self, src):
        if not isinstance(src, Source):
            return TypeError('src must be a Source instance')
        return self.pixels == src.pixels
                            
    def __len__(self):
        return len(self.pixels)
        
    def __iter__(self):
        return iter(self.pixels)
    
    def __str__(self):
        return 'Source(' + ','.join([str(pix) for pix in self.pixels]) + ')'
        

    
    def add(self, pix, clip=False, ctype=int):
        """
        :param pix: must be a tuple of singular numerical values beginning
        with the coordinates of the sources: (x,y,...)

        """
        pix = Pixel(pix, ctype=ctype) 

        if self.pixel_size is not None:
            if self.pixel_size != len(pix):
                raise TypeError('pix must have len {}'.format(self.pixel_size))
        else: self.pixel_size = len(pix)
            
        if self.check_pixels:
            if not pix.validate(0, self.maxx, 0, self.maxy):
                if clip: return
                else: raise ValueError('pixel at {} is off boundaries (0:{}, 0:{})'.format(
                        pix, self.maxx, self.maxy))
        
        self.pixels.add(pix)

    def remove(self, pix):
        src = Source(pix)
        for ipix in src:
            if ipix in self.pixels:
                self.pixels.remove(ipix)
        
    def update(self, pixs, clip=False, ctype=int):
        type_ok = False
        for itype in (Source, set):
            if isinstance(pixs, itype): type_ok = True
        if not type_ok:
            raise TypeError('pixs is {} but must be a set or a Source'.format(type(pixs)))
        for ipix in pixs:
            self.add(ipix, clip=clip, ctype=ctype)
            
    def get_neighbors(self):
        nns = Source(frame_shape=self.shape)
        for ipix in self:
            nns.update(ipix.get_neighbors(), clip=True)
        nns.remove(self)
        return nns

    def get_pixels_in_radius(self, radius):
        nns = Source(frame_shape=self.shape)
        for ipix in self:
            nns.update(ipix.get_neighbors_in_radius(radius), clip=True)
        return nns
    
    def get_neighbors_in_radius(self, radius):
        nns = self.get_pixels_in_radius(radius)
        nns.remove(self)
        return nns

    def counts(self):
        return len(self)
    
    def centroid(self):
        return Pixel(tuple(np.mean(self.to_array()[:,:2], axis=0)), ctype=int)

    def mean(self):
        return list(np.mean(self.to_array(), axis=0)[2:])

    def sum(self):
        return list(np.sum(self.to_array(), axis=0)[2:])
    
    def to_array(self):
        return np.array([pix.data for pix in self.pixels])
    
    def to_indices(self):
        arr = self.to_array().astype(int)
        if np.size(arr) > 0:
            return (arr[:,0], arr[:,1])
        else:
            return ()

    def astype(self, ctype):
        src = Source()
        for ipix in self:
            src.add(ipix.astype(ctype))
        return src

    def validate(self, xmin, xmax, ymin, ymax):
        bad_pixels = set()
        for ipix in self:
            if not ipix.validate(xmin, xmax, ymin, ymax):
                bad_pixels.add(ipix)
        self.remove(bad_pixels)
        return self

#################################################
#### CLASS Sources ##############################
#################################################
    
class Sources(object):

    def __init__(self, data=None):

        self.sources = list()
        self.pixel_size = None
        

        if data is not None:
            if isinstance(data, str):
                # load hdf5 sources file
                self.sources = list()
                file_valid = True
                with orb.utils.io.open_hdf5(data, 'r') as fin:
                    index = 0
                    while self._get_hdf_source_path(index) in fin:
                        isrc = Source()
                        isrc.update(
                            set([tuple(ipix) for ipix
                                 in fin[self._get_hdf_source_path(index)][:]]))
                        self.append(isrc)
                        index += 1

                    if index == 0: file_valid = False

                    if not file_valid: raise TypeError('invalid hdf5 file')
                logging.info('{} sources loaded'.format(len(self.sources)))
            else:
                try:
                    self.append(data)
                except TypeError:
                    try:
                        self.update(data)
                    except TypeError:
                        raise TypeError('data type not understood')

    def __len__(self):
        return len(self.sources)

    def __str__(self):
        return 'Sources(' + ',\n  '.join([str(src) for src in self.sources]) + ')'

    def __iter__(self):
        return iter(self.sources)

    def __getitem__(self, index):
        return self.sources[index]

    def _get_hdf_source_path(self, index):
        return '/source{:05d}'.format(index)
    
    def append(self, src):
        """Append a source only if it is not already present in the list
        """
        src = Source(src)
        if self.pixel_size is not None:
            if self.pixel_size != src.pixel_size:
                raise TypeError('src element size must have size {}'.format(
                    src.pixel_size))
        else: self.pixel_size = src.pixel_size

        if src not in self:
            self.sources.append(src)

    def pop(self, index):
        index = int(index)
        return self.sources.pop(index)

    def update(self, sources):
        sources = list(sources)
        for isrc in sources:
            self.append(isrc)
            
    def get_default_hdf_path(self):
        return './sources.hdf5'
    
    def to_hdf(self, path=None):
        if path is None: path = self.get_default_hdf_path()
        with orb.utils.io.open_hdf5(path, 'w') as fout:
            for isrc in range(len(self.sources)):
                fout[self._get_hdf_source_path(isrc)] = self.sources[isrc].to_array()

    def centroid(self):
        return [src.centroid().xy for src in self]

    def mean(self):
        return [src.mean() for src in self]

    def sum(self):
        return [src.sum() for src in self]

    def counts(self):
        return [src.counts() for src in self]

    def distance(self, pix):
        pix = Pixel(pix)
        cent = np.array(self.centroid())
        return np.sqrt((cent[:,0] - pix.x)**2 + (cent[:,1] - pix.y)**2)
    
    def xmatch(self, sources, radius=1):
        xm_self = Sources()
        xm_sources = Sources()
        nonxm_sources = Sources(sources)
        nonxm_self = Sources()

        for isrc in self:
            idist = nonxm_sources.distance(isrc.centroid())
            if np.min(idist) <= radius:
                xm_self.append(isrc)
                xm_sources.append(
                    nonxm_sources.pop(np.argmin(idist)))
            else:
                nonxm_self.append(isrc)

        logging.info('number of xmatchs: {}'.format(len(xm_self)))
        logging.info('number of non-xmatchs in self: {}/{}'.format(
            len(nonxm_self), len(self)))
        logging.info('number of non-xmatchs in sources: {}/{}'.format(
            len(nonxm_sources), len(sources)))
        
        return xm_self, nonxm_self, xm_sources, nonxm_sources

                
    def validate(self, xmin, xmax, ymin, ymax):
        for isrc in self:
            isrc.validate(xmin, xmax, ymin, ymax)
        return self
    

    
#################################################
#### CLASS Aggregator ###########################
#################################################
    
class Aggregator(object):
    
    MAX_SOURCE_SIZE = 1000
    MAX_SOURCES = 10000
    
    def __init__(self, frame, aggregation_min, detection_min):
        orb.utils.validate.is_2darray(frame, object_name='frame')
        if not isinstance(aggregation_min, float): raise TypeError('aggregation_min must be a float')
        if not isinstance(detection_min, float): raise TypeError('detection_min must be a float')
        
        self.frame = np.copy(frame)
        self.shape = tuple(self.frame.shape)
        self.mask = np.ones_like(self.frame, dtype=bool)
        self.mask[np.isnan(self.frame)] = False
        self.frame[np.isnan(self.frame)] = 0
        self.detection_min = float(detection_min)
        self.aggregation_min = float(aggregation_min)
        self.sources = Sources()
    
    def _set_detection_threshold(self):
        dt = np.max(self.frame * self.mask)
        if dt >= self.detection_min: self.detection_threshold = float(dt)
        else: self.detection_threshold = None
        return self.detection_threshold

    def _get_next_max(self):
        return np.unravel_index(np.argmax(self.frame * self.mask >= self.detection_threshold), self.shape)

    def run(self, max_source_size=None):
        if max_source_size is None:
            max_source_size = self.MAX_SOURCE_SIZE
            
        progress = orb.core.ProgressBar(0)
        while self._set_detection_threshold() is not None:
            isrc = Source(frame_shape=self.shape)
            nextmax = self._get_next_max()
            isrc.add((nextmax[0], nextmax[1], self.frame[nextmax]))
            grow = True
            while grow:
                grow = False
                count = int(len(isrc))
                
                for inei in isrc.get_neighbors():
                    if self.frame[inei.xy] >= self.aggregation_min:
                        isrc.add((inei.x, inei.y, self.frame[inei.xy]))
                        
                if len(isrc) >= max_source_size:
                    break
                                            
                if len(isrc) > count:
                    grow = True
            for iisrc in isrc:
                self.mask[iisrc.xy] = False
                
            if len(isrc) >= max_source_size:
                warnings.warn('source rejected because its size is > {}'.format(
                    max_source_size))
            else:    
                self.sources.append(isrc)
                progress.update(0, info='number of sources detected: {}'.format(len(self.sources)))
            
            if len(self.sources) > self.MAX_SOURCES:
                warnings.warn('max sources limit reached')
                break
            
        logging.info('number of sources detected: {}'.format(len(self.sources)))

        return Sources(self.sources)
    
    def get_sources(self):
        return Sources(self.sources)
    
    

