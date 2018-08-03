#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: nn.py

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
ORCS (Outils de Réduction de Cubes Spectraux) provides tools to
extract data from ORBS spectral cubes.

This module contains the fitting classes
"""
import logging
import numpy as np
import time

import orcs.core
import orcs.utils
import orb.utils.io
import orb.utils.spectrum
import orb.cutils
import orb.core
import orb.utils.validate
import orb.utils.image

import tensorflow as tf

class Background(orb.core.Tools):

    ROI_BORDER_FACTOR = 0.05
    BATCH_SIZE = 1000
    BBOX_FACTOR = 3
    
    def __init__(self, shape, data_prefix=None):
        
        orb.utils.validate.is_iterable(shape, object_name='shape')
        self.shape = np.array(shape).astype(int)

        self.dimx = int(self.shape[0])
        if len(shape) > 1:
            if int(self.shape[1])&1: raise ValueError('shape[1] must be even')
            self.dimy = int(self.shape[1])
        if len(shape) > 2:
            if int(self.shape[2])&1: raise ValueError('shape[2] must be even')
            self.dimz = int(self.shape[2])
        

        data_suffix = '.'.join(self.shape.astype(str))
        if data_prefix is None:
            data_prefix = self.__class__.__name__ + '.' + data_suffix + '.'
            
        elif not isinstance(data_prefix, str):
            raise TypeError('data_prefix must be a string')

        self.data_prefix = orcs.core.DataFiles().get_path(data_prefix)
        self.load_dataset()


    def get_hdf_batch_name(self, index):
        return '/batch_{:05d}'.format(index)
    
    def get_hdf_batch_index(self, name):
        return int(name.split('_')[1])

    def load_dataset(self):
        with orb.utils.io.open_hdf5(
                self.get_backgrounds_file_path(), 'r') as f:
            self.batch_nb = 0
            self.box_nb = 0
            for iname in f:
                iindex = self.get_hdf_batch_index(iname)
                if self.batch_nb <= iindex:
                    self.batch_nb = iindex + 1
                self.box_nb += f[iname].shape[0]
            logging.debug('batch number: {}'.format(self.batch_nb))
            logging.debug('box number: {}'.format(self.box_nb))

    def get_backgrounds_file_path(self):
        return self.data_prefix + 'dataset.hdf5'

    def extract(self, cube_path, batch_number, reset=False):
        cube = orcs.core.HDFCube(cube_path)
        zlim = cube.get_filter_range_pix()
        zlim[0] += self.ROI_BORDER_FACTOR * cube.dimz
        zlim[1] -= self.ROI_BORDER_FACTOR * cube.dimz + self.dimz * self.BBOX_FACTOR
        
        xlim = self.ROI_BORDER_FACTOR * cube.dimx, (1 - self.ROI_BORDER_FACTOR) * cube.dimx - self.dimx * self.BBOX_FACTOR
        ylim = self.ROI_BORDER_FACTOR * cube.dimy, (1 - self.ROI_BORDER_FACTOR) * cube.dimy - self.dimy * self.BBOX_FACTOR

        if reset:
            open_mode = 'w'
            self.batch_nb = 0
        else: open_mode = 'a'
        
        with orb.utils.io.open_hdf5(self.get_backgrounds_file_path(), open_mode) as f:

            first_index = int(self.batch_nb)
            logging.debug('first index: {}'.format(first_index))

            boxes = list()
            for ibatch in range(batch_number):
                progress = orb.core.ProgressBar(self.BATCH_SIZE * self.BBOX_FACTOR**3)
                batch = list()
                for ii in range(self.BATCH_SIZE):
                    if not ii % (self.BATCH_SIZE // 100):
                        progress.update(ii*self.BBOX_FACTOR**3, info='extracting batch {}/{}: {}/{}'.format(
                            ibatch, batch_number,
                            ii*self.BBOX_FACTOR**3,
                            self.BATCH_SIZE*self.BBOX_FACTOR**3))

                    valid = False
                    while not valid:
                        xr = int(np.random.uniform(xlim[0], xlim[1]))
                        yr = int(np.random.uniform(ylim[0], ylim[1]))
                        zr = int(np.random.uniform(zlim[0], zlim[1]))
                        bbox = cube.get_data(xr, xr + self.dimx * self.BBOX_FACTOR,
                                             yr, yr + self.dimy * self.BBOX_FACTOR,
                                             zr, zr + self.dimz * self.BBOX_FACTOR,
                                             silent=True)
                        if not np.any(np.isnan(bbox)): valid = True

                    iboxes = list()
                    for iib in range(self.BBOX_FACTOR):
                        for ijb in range(self.BBOX_FACTOR):
                            for ikb in range(self.BBOX_FACTOR):
                                ibox = bbox[iib*self.dimx:(iib + 1)*self.dimx,
                                            ijb*self.dimy:(ijb + 1)*self.dimy,
                                            ikb*self.dimz:(ikb + 1)*self.dimz]
                        
                                ibox -= np.median(ibox, axis=[0,1])
                                ibox /= np.std(ibox[(ibox < np.percentile(ibox, 98))
                                                    * (ibox > np.percentile(ibox, 2))])
                                iboxes.append(ibox)
                    batch += iboxes
                progress.end()
                f.create_dataset(
                    self.get_hdf_batch_name(ibatch + first_index),
                    data=np.array(batch))
                
        self.load_dataset()
    
    def get_batch(self, index):
        if index >= self.batch_nb: raise ValueError('index must be < {}'.format(self.batch_nb))
        with orb.utils.io.open_hdf5(
                self.get_backgrounds_file_path(), 'r') as f:
            return f[self.get_hdf_batch_name(index)][:]

    def get_random_batch(self):
        return self.get_batch(np.random.randint(self.batch_nb))

    def get_random_box(self):
        rbatch = np.random.randint(self.batch_nb)
        with orb.utils.io.open_hdf5(
                self.get_backgrounds_file_path(), 'r') as f:
            rbox = np.random.randint(f[self.get_hdf_batch_name(rbatch)].shape[0])
            outbox = f[self.get_hdf_batch_name(rbatch)][rbox, ...]
            # returned box is flipped randomly
            irot = np.random.randint(4)
            if irot == 0: return outbox
            elif irot == 1: return np.flip(outbox, 0)
            elif irot == 2: return np.flip(outbox, 1)
            elif irot == 3: return np.flip(np.flip(outbox, 0), 1)
            else: raise ValueError('bad rotation type')


class NNWorker(orb.core.Tools):

    TRAIN_RATIO = 0.8
    BATCH_SIZE = 1000
    MAX_SURFACE_COEFF = 0.05

    def __init__(self, shape, data_prefix=None):
        if data_prefix is None:
            data_prefix = self.__class__.__name__ + '.'
        elif not isinstance(data_prefix, str):
            raise TypeError('data_prefix must be a string')

        self.data_prefix = orcs.core.DataFiles().get_path(data_prefix)

        orb.utils.validate.is_iterable(shape, object_name='shape')
        self.shape = np.array(shape).astype(int)

        if int(self.shape[0])&1: raise ValueError('shape[0] must be even')
        self.dimx = int(self.shape[0])
        if len(shape) > 1:
            if int(self.shape[1])&1: raise ValueError('shape[1] must be even')
            self.dimy = int(self.shape[1])
        if len(shape) > 2:
            if int(self.shape[2])&1: raise ValueError('shape[2] must be even')
            self.dimz = int(self.shape[2])
        if self.TRAIN_RATIO >= 1:
            raise ValueError('TRAIN_RATIO must be < 1')

        self.dataset = None

    def get_session(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        return self.session

    def init_session(self, is_training=False):
        if is_training:
            if self.dataset is None:
                raise StandardError('dataset must be loaded first')
            
        self.generate_graph()
        self.get_session()
        if not is_training:
            self.saver.restore(self.session, self.get_model_path())
        else:
            self.init.run(session=self.session)
        self.nclasses = int(self.proba.shape[-1]) # get number of output classes
        return self.session

    def save_graph(self):
        self.saver.save(self.session, self.get_model_path())

    def simulate(self):
        pass

    def generate_graph(self):
        """Must instanciate self.saver, self.init, self.accuracy,
        self.training_op, self.proba"""
        pass

    def get_dataset_path(self):
        return self.data_prefix + 'dataset'

    def get_model_path(self):
        return self.data_prefix + 'model.ckpt'

    def load_dataset(self, force=True):

        if self.dataset is not None and not force:
            logging.warning('dataset already loaded')
            return

        self.dataset = dict()

        with orb.utils.io.open_hdf5(self.get_dataset_path(), 'r') as inf:
            if 'data' not in inf.keys(): raise ValueError('data must be in dataset')
            if 'label' not in inf.keys(): raise ValueError('label must be in dataset')
            size = inf['data'].shape[0]
            train_slice = slice(0, int(size * self.TRAIN_RATIO))
            valid_slice = slice(train_slice.stop, size)
            
            logging.info('dataset size: {}'.format(size))
            for ikey in inf.keys():
                idat = inf[ikey][:]
                if np.any(np.isnan(idat)): raise Exception('nan in data')
                if np.any(np.isinf(idat)): raise Exception('inf in data')
                train_key = ikey + '_train'
                valid_key = ikey + '_valid'
                
                self.dataset[train_key] = np.squeeze(idat[train_slice,:])
                self.dataset[valid_key] = np.squeeze(idat[valid_slice,:])
                
                logging.debug('{} shape: {}'.format(
                    train_key, self.dataset[train_key].shape))
                logging.debug('{} shape: {}'.format(
                    valid_key, self.dataset[valid_key].shape))
                
    def generate_dataset(self, size, rotate=True, noise_samples=10):
        if not isinstance(size, int): raise TypeError('size must be an int')
        if not size > 0: raise ValueError('size must be > 0')

        dataset = dict()

        if rotate:
            rotation_nb = 4
        else:
            rotation_nb = 1

        loopsize = size // rotation_nb // noise_samples

        progress = orb.core.ProgressBar(loopsize)
        first_sim = True
        for isim in range(loopsize):
            if not isim%1000:
                progress.update(isim, info='generating dataset')
            sim_list = self.simulate(samples=noise_samples)
            for sim_dict in sim_list:

                for irot in range(rotation_nb):
                    rotated_sim_dict = dict(sim_dict)
                    if irot == 1:
                        rotated_sim_dict['data'] = np.flip(sim_dict['data'], 0)
                    elif irot == 2:
                        rotated_sim_dict['data'] = np.flip(sim_dict['data'], 1)
                    elif irot == 3:
                        rotated_sim_dict['data'] = np.flip(np.flip(sim_dict['data'], 0), 1)

                    for ikey in sim_dict:
                        if first_sim:
                            if ikey in dataset.keys(): raise ValueError(
                                '{} already appended'.format(ikey))
                            dataset[ikey] = list()
                        dataset[ikey].append(np.array(rotated_sim_dict[ikey]).flatten())
                    first_sim = False
        progress.end()

        path = self.get_dataset_path()
        with orb.utils.io.open_hdf5(path, 'w') as outf:
            for ikey in dataset:
                idata = np.array(dataset[ikey])
                outf.create_dataset(ikey, data=idata)
                logging.info('{} shape: {}'.format(
                    ikey, idata.shape))
        return path


    def get_shuffler(self, sorting_dict=None,
                     datatype='train'):
            """
            :param sorting_dict: must be a dictionary giving for each label its
              sorting key (e.g. {1:'snr'} to sort label 1 by snr (), if a 'snr'
              is provided in the dataset).
            """
            if sorting_dict is not None:
                if not isinstance(sorting_dict, dict):
                    raise TypeError('sorting_dict must be a dict')

            rnd_idx = np.random.permutation(len(self.dataset['data_' + datatype]))

            if sorting_dict is not None:
                for ilabel in sorting_dict:
                    # get indexes corresponding to the label in the permuted dataset
                    label_idx = np.nonzero(
                        self.dataset['label_' + datatype][rnd_idx] == ilabel)[0]
                    # sort indexes corresponding to the label against the set
                    # specified in sorting_dict (high values first)
                    sorted_label_idx = label_idx[np.argsort(
                        self.dataset[sorting_dict[ilabel]
                                     + '_' + datatype][rnd_idx][label_idx])[::-1]]
                    # replace the unsorted indexes corresponding to the label with
                    # the sorted ones
                    rnd_idx[label_idx] = rnd_idx[sorted_label_idx]

            n_batches = len(self.dataset['data_' + datatype]) // self.BATCH_SIZE
            for batch_idx in np.array_split(rnd_idx, n_batches):
                if sorting_dict is not None:
                    # each batch is shuffled again because some labels have been
                    # sorted in the batch
                    np.random.shuffle(batch_idx)

                X_batch = self.dataset['data_' + datatype][batch_idx]
                y_batch = self.dataset['label_' + datatype][batch_idx]
                yield X_batch, y_batch


    def get_training_shuffler(self, sorting_dict=None):
        return self.get_shuffler(
            sorting_dict=sorting_dict,
            datatype='train')

    def get_validation_shuffler(self, sorting_dict=None):
        return self.get_shuffler(
            sorting_dict=sorting_dict,
            datatype='valid')
                
    def train(self, n_epochs=4, sorting_dict=None):

        LOOP_UPDATE = 30
        
        output = dict()
        output['train_loss'] = list()
        output['train_accu'] = list()
        output['valid_loss'] = list()
        output['valid_accu'] = list()
        
        with self.init_session(is_training=True):

            for epoch in range(n_epochs):
                progress = orb.core.ProgressBar(
                    len(self.dataset['data_train']) // self.BATCH_SIZE)
                iloop = 1
                training_shuffler = self.get_training_shuffler()
                for X_batch, y_batch in training_shuffler:
                    # train on batch                    
                    _, tloss, tacc = self.session.run(
                        [self.training_op, self.loss, self.accuracy],
                        feed_dict={
                            self.X: X_batch, self.y: y_batch,
                            self.is_training: True})
                    output['train_loss'].append(tloss)
                    output['train_accu'].append(tacc)

                    # get validation loss                    
                    X_valid_batch, y_valid_batch = self.get_validation_shuffler().next()
                    vloss, vacc = self.session.run(
                        [self.loss, self.accuracy],
                        feed_dict={
                            self.X: X_valid_batch, self.y: y_valid_batch,
                            self.is_training:False})
                    output['valid_loss'].append(vloss)
                    output['valid_accu'].append(vacc)

                    if not iloop%LOOP_UPDATE and iloop>1:
                        progress.update(
                            iloop, info='loss: {:.2e}|{:.2e}, acc: {:.2e}|{:.2e}'.format(
                                np.median(output['train_loss'][:-LOOP_UPDATE]),
                                np.median(output['valid_loss'][:-LOOP_UPDATE]),
                                np.median(output['train_accu'][:-LOOP_UPDATE]),
                                np.median(output['valid_accu'][:-LOOP_UPDATE])))

                    
                    iloop += 1
                progress.end()

                self.save_graph()
                logging.info('model saved as {}'.format(self.get_model_path()))
                
        self.validate()
        
        return output


    def validate(self):
        with self.init_session(is_training=False):
            # check validation set accuracy
            stime = time.time()
            acc_list = list()
            loss_list = list()
            for X_batch, y_batch in self.get_validation_shuffler():
                vloss, vacc = self.session.run(
                    [self.loss, self.accuracy],
                    feed_dict={
                        self.X: X_batch, self.y: y_batch,
                        self.is_training: False})
                acc_list.append(vacc)
                loss_list.append(vloss)
            logging.info('validation: time: {:.1f}s | loss: {:.3f} [{:.3f}] | accuracy: {:.3f} [{:.3f}]'.format(
                time.time() - stime,
                np.median(loss_list), np.std(loss_list),
                np.median(acc_list), np.std(acc_list)))


    # def test(self):
    #     self.load_dataset()
    #     predictions, probas = self.run_on_sample(self.dataset['data_test'])

    #     diff = (predictions - self.dataset['label_test']) != 0
    #     logging.info('number of predictions: {}'.format(diff.size))
    #     logging.info('number of errors: {}'.format(np.sum(diff)))

    #     return predictions, probas

    def run_on_sample(self, sample):
        """Note that there is no standardization of the sample.
        """
        sample = np.array(sample)
        with self.init_session():

            probas_list = list()
            predictions_list = list()
            
            for ibatch in range(0, sample.shape[0], self.BATCH_SIZE):
                X_batch = sample[ibatch:ibatch+self.BATCH_SIZE, ...]
                probas_list.append(np.array(self.proba.eval(
                    feed_dict={self.X: X_batch, self.is_training: False})))
                predictions_list.append(
                    np.array(tf.argmax(probas_list[-1], axis=1).eval()))

            predictions = np.concatenate(predictions_list)
            probas = np.concatenate(probas_list)

        return predictions, probas

    def _run_on_cube(self, cube, zlimits=None, slices=None, record=False):
        """cube can be an np.ndarray or an HDFCube instance

        :param slices: A list of slices [[slice(xmin, xmax), slice(ymin, ymax)],
          ...]. Note that there is a limit on the number of slices that can be
          passed.
        """

        def box_streamer(box, dimz):
            for ik in range(box.shape[2] - dimz):
                yield box[..., ik:ik+dimz].flatten()

        def get_slices():
            return orcs.utils.image_streamer(
                cube.shape[0], cube.shape[1], [self.dimx, self.dimy])

        SLICES_NB_THRESHOLD = 2e5 # max number of slices
        HIGHEST_NB = 20
        SKYBOX_SIZE = 2 # must be even 

        if slices is None:
            slices_nb = ((cube.shape[0] - self.dimx)
                         * (cube.shape[1] - self.dimy))
            slices = get_slices()
        else:
            orb.utils.validate.is_iterable(slices, object_name='slices')
            slices_nb = len(slices)

        if slices_nb > SLICES_NB_THRESHOLD:
            raise MemoryError('too much slices: {} > {}'.format(
                slices_nb, SLICES_NB_THRESHOLD))

        
        with self.init_session():

            # init output dict
            output = dict()
            output['proba_max'] = list()
            output['proba_median'] = list()
            output['proba_min'] = list()
            output['proba_argmax'] = list()
            # output['proba_highest'] = list()
            # output['proba_arghighest'] = list()
            # output['proba_highest_sum'] = list()
            if record:
                timestamp = orcs.utils.get_timestamp()
                output['proba_path'] = list()

            # run detection
            progress = orb.core.ProgressBar(slices_nb)
            iloop = 0
            for islice in slices:
                if not iloop%100:
                    progress.update(iloop, 'running on slice {}/{}'.format(iloop, slices_nb))

                if zlimits is None:
                    idata_box = np.copy(cube[islice[0], islice[1], :])
                    zmin = 0

                else:
                    idata_box = np.copy(cube[islice[0], islice[1], zlimits[0]:zlimits[1]])
                    zmin = zlimits[0]
                                                            
                isky_box = np.copy(idata_box)
                isky_box[self.dimx//2 - SKYBOX_SIZE//2:
                         self.dimx//2 + SKYBOX_SIZE//2,
                         self.dimy//2 - SKYBOX_SIZE//2:
                         self.dimy//2 + SKYBOX_SIZE//2] = np.nan

                # standardization
                idata_box -= np.nanmedian(isky_box, axis=[0,1])
                idata_box /= np.median(np.nanstd(isky_box, axis=(0,1)))

                idata_box[np.nonzero(np.isnan(idata_box))] = 0.
                                
                boxes = list()
                for ibox in box_streamer(idata_box, self.dimz):
                    boxes.append(ibox)
                boxes = np.array(boxes)

                probas_list = list()
                for ibatch in range(0, boxes.shape[0], self.BATCH_SIZE):

                    X_batch = boxes[ibatch:ibatch+self.BATCH_SIZE, ...]
                    probas_list.append(np.array(self.proba.eval(
                        feed_dict={self.X: X_batch, self.is_training: False})))

                ## probas must be outputed for each label !!
                iprobas = np.concatenate(probas_list)

                iprobas_sorted = np.argsort(iprobas, axis=0)
                output['proba_argmax'].append(iprobas_sorted[-1,:] + zmin)
                output['proba_max'].append(
                    np.diag(iprobas[iprobas_sorted[-1,:]]))
                output['proba_median'].append(
                    np.diag(iprobas[iprobas_sorted[iprobas.shape[0]//2,:]]))
                output['proba_min'].append(
                    np.diag(iprobas[iprobas_sorted[0,:]]))
                # output['proba_arghighest'].append(
                #     iprobas_sorted[-HIGHEST_NB:,:] + zmin)
                # output['proba_highest'].append(
                #     np.diagonal(iprobas[iprobas_sorted[-HIGHEST_NB:,:]],
                #                 axis1=2))
                # output['proba_highest_sum'].append(
                #     np.sum(output['proba_highest'][-1], axis=0))

                if record:
                    iproba_paths = list()
                    for iclass in range(self.nclasses):
                        iproba_path = './{}/proba.{}.{}.fits'.format(
                            timestamp, iclass, iloop)
                        iproba_paths.append(iproba_path)
                        orb.utils.io.write_fits(
                            iproba_path,
                            iprobas[:, iclass], overwrite=True)
                    output['proba_path'].append(iproba_paths)

                iloop += 1
            progress.end()

        formatted_output = list()
        for iclass in range(self.nclasses):
            formatted_output.append(dict())
            for ikey in output:
                output[ikey] = np.array(output[ikey])
                formatted_output[iclass][ikey] = list(output[ikey][...,iclass])
        return formatted_output


    def run_on_cube_targets(self, cube, targets, record=False, aper=0):

        cube._silent_load = True
        zlimits = np.round(np.array(cube.get_filter_range_pix())).astype(int)
        zlimits[0] += cube.dimz * 0.02
        zlimits[1] -= cube.dimz * 0.02
        
        logging.debug('computation restricted to filter limits: {}'.format(zlimits))
        
        targets = np.round(np.array(targets))
        if targets.shape[1] != 2:
            raise TypeError('badly formatted target list. Must have shape (n, 2) but has shape {}'.format(targets.shape))

        # check targets
        xmin = orb.utils.validate.index(
            targets[:,0] - self.dimx//2, aper * 2, cube.dimx - self.dimx - aper * 2, clip=False)
        xmax = orb.utils.validate.index(
            targets[:,0] + self.dimx//2, aper * 2, cube.dimx - aper * 2, clip=False)
        ymin = orb.utils.validate.index(
            targets[:,1] - self.dimy//2, aper * 2, cube.dimy - self.dimy - aper * 2, clip=False)
        ymax = orb.utils.validate.index(
            targets[:,1] + self.dimy//2, aper * 2, cube.dimy - aper * 2, clip=False)

        slices = list()
        for i in range(targets.shape[0]):
            for iaper in range(-(aper), aper+1):
                for japer in range(-(aper), aper+1):
                    slices.append([slice(xmin[i] + iaper, xmax[i] + iaper),
                                   slice(ymin[i] + japer, ymax[i] + japer)])

        output = self._run_on_cube(cube, zlimits=zlimits, slices=slices,
                                   record=record)

        for iout in output:
            for ikey in iout:
                if isinstance(iout[ikey][0], str):
                    iout[ikey] = [isplit[((aper*2)+1 ** 2)//2] for isplit
                                  in np.split(np.array(iout[ikey]), len(targets))]
                else:
                    iout[ikey] = np.mean(np.split(np.array(iout[ikey]), len(targets)),
                                         axis=1)

        return output


    def run_on_cube_subregion(self, cube, start, stop, record=True):

        MAX_DATA_SURFACE = self.MAX_SURFACE_COEFF * cube.dimx * cube.dimy
        
        # validate input
        orb.utils.validate.has_len(start, 2, object_name='start')
        orb.utils.validate.has_len(stop, 2, object_name='start')

        orb.utils.validate.index(start[0] - self.dimx//2, 0, cube.dimx, clip=False)
        orb.utils.validate.index(stop[0] + self.dimx//2, 0, cube.dimx, clip=False)
        orb.utils.validate.index(start[1] - self.dimy//2, 0, cube.dimy, clip=False)
        orb.utils.validate.index(stop[1] + self.dimy//2, 0, cube.dimy, clip=False)

        # prepare data
        data_xrange = range(start[0] - self.dimx//2, stop[0] + self.dimx//2 + 1)
        data_yrange = range(start[1] - self.dimy//2, stop[1] + self.dimy//2 + 1)

        zlimits = np.array(cube.get_filter_range_pix()).astype(int)

        data_surface = len(data_xrange) * len(data_yrange)
        if data_surface > MAX_DATA_SURFACE:
            raise MemoryError('subregion surface is {} and must be smaller than {}'.format(
                data_surface, MAX_DATA_SURFACE))

        # return ({'test':np.zeros((stop[0] - start[0], stop[1] - start[1], 1))},
        #         {'test':np.zeros((stop[0] - start[0], stop[1] - start[1], 1))})

        idata_slice = cube.get_data(data_xrange[0], data_xrange[-1],
                                    data_yrange[0], data_yrange[-1],
                                    zlimits[0],
                                    zlimits[1] - self.dimz,
                                    silent=False)

        output = self._run_on_cube(idata_slice, zlimits=None)

        timestamp = orcs.utils.get_timestamp()
        for iclass in range(len(output)):
            for ikey in output[iclass]:
                output[iclass][ikey] = np.reshape(
                    output[iclass][ikey],
                    (stop[0] - start[0], stop[1] - start[1], -1))
                if record:
                    orb.utils.io.write_fits(
                        './{}/outmap.{}.{}.fits'.format(timestamp, iclass, ikey),
                        output[iclass][ikey], overwrite=True)

        return output

    def run_on_cube(self, cube):

        max_size = int(np.sqrt(self.MAX_SURFACE_COEFF * 0.9) * min(cube.dimx, cube.dimy))
        output = list()
        timestamp = orcs.utils.get_timestamp()
        x_range = range(self.dimx//2, cube.dimx - self.dimx, max_size)
        y_range = range(self.dimy//2, cube.dimy - self.dimy, max_size)
        isubregion = 0
        stime = time.time()
        for ii in x_range:
            for ij in y_range:
                isubregion += 1
                start = [ii, ij]
                
                stop = [min(ii + max_size,
                            cube.dimx - self.dimx//2 - 1),
                        min(ij + max_size,
                            cube.dimy - self.dimy//2 - 1)]
                if np.any(np.array(stop) - np.array(start)
                          - np.array([self.dimx, self.dimy]) < 0): continue
                
                logging.info('working on subregion {}/{} [{}:{},{}:{}] (remains: {:.2f}h)'.format(
                    isubregion, len(x_range) * len(y_range),
                    start[0], stop[0], start[1], stop[1],
                    (time.time() - stime) / isubregion * len(x_range) * len(y_range)))
                iout = self.run_on_cube_subregion(cube, start, stop, record=False)
                for iclass in range(len(iout)):
                    if len(output) <= iclass: output.append(dict())
                    for ikey in iout[iclass]:
                        if ikey not in output[iclass]:
                            if np.squeeze(iout[iclass][ikey]).ndim != 2:
                                raise TypeError('expected output is a 2d map')
                            output[iclass][ikey] = np.empty((cube.dimx, cube.dimy),
                                                            dtype=float)
                            output[iclass][ikey].fill(np.nan)
                        output[iclass][ikey][start[0]:stop[0],
                                             start[1]:stop[1]] = np.squeeze(
                                                 iout[iclass][ikey])
                        orb.utils.io.write_fits(
                            './{}/outmap.{}.{}.fits'.format(timestamp, iclass, ikey),
                            output[iclass][ikey], overwrite=True)

                    
        
            

class SourceDetector3d(NNWorker):

    DIMX = 8  # x size of the box
    DIMY = 8 # y size of the box
    DIMZ = 8  # number of channels in the box
    OVERSAMPLING_RATIO = [1.2, 1.3] # total step_nb / step_nb at the right of ZPD
    SINC_SIGMA = [0, 0.5] # sinc broadening 
    SOURCE_BETA = [3., 4.] # source moffat beta parameter
    SOURCE_FWHM = [2.5, 4]  # SINC FWHM = 1.20671 * WIDTH
    #TIPTILT = np.pi / 8
    CHANNEL_R = 1 # uniform distribution of channel
    POS_R = 1 # uniform distribution of the position
    SNR = (1, 30)  # min max SNR (log uniform)

    # hyperparameters
    FILTERS = 8
    LEARNING_RATE = 0.01 # 0.001 by default for AdamOptimizer
    EPSILON = 1e-8 # 1e-8 by default for AdamOptimizer
    BETA1 = 0.9 # 0.9
    BETA2 = 0.999 # 0.999
    OUTPUT_DROPOUT_RATE = 0.97 # dropout rate of output layer 
    INPUT_DROPOUT_RATE = 0. # dropout rate of input unit (better if close to 1) (Srivastava 2014)
    KERNEL_SIZE = 2
    
    def __init__(self):

        NNWorker.__init__(self, (self.DIMX, self.DIMY, self.DIMZ))
        self.background = Background(self.shape)
        
    def simulate(self, snr=None, samples=1):

        # simulate source
        src3d = np.zeros(self.shape, dtype=np.float32)
 
        if snr is None:
            has_source = np.random.randint(0, 2)
            if has_source: # log uniform distribution of the SNR
                snr = 10**(np.random.uniform(
                    np.log10(self.SNR[0]),
                    np.log10(self.SNR[1])))
                #snr = np.random.uniform(self.SNR[0], self.SNR[1])
            else:
                snr = 0.
        else:
            has_source = 1
            snr = float(snr)        

        if has_source:
            # emission line source
            dxr = self.dimx // 2. - 0.5 + np.random.uniform(-self.POS_R, self.POS_R)
            dyr = self.dimy // 2. - 0.5 + np.random.uniform(-self.POS_R, self.POS_R)
            source_fwhmr = np.random.uniform(self.SOURCE_FWHM[0], self.SOURCE_FWHM[1])
            source_betar = np.random.uniform(self.SOURCE_BETA[0], self.SOURCE_BETA[1])
            src2d = orb.cutils.moffat_array2d(
                0, 1,
                dxr, dyr,
                source_fwhmr,
                source_betar,
                self.dimx, self.dimy)
            src2d = src2d.reshape((self.dimx, self.dimy, 1))

            # simulate emission line spectrum
            sinc_fwhm_limits = orb.utils.spectrum.compute_line_fwhm_pix(
                oversampling_ratio=np.array(self.OVERSAMPLING_RATIO))

            sinc_fwhmr = np.random.uniform(sinc_fwhm_limits[0],
                                           sinc_fwhm_limits[1])
            sinc_sigmar = np.random.uniform(self.SINC_SIGMA[0],
                                            self.SINC_SIGMA[1])
            
            channelr = (np.random.uniform(-self.CHANNEL_R, self.CHANNEL_R)
                        + self.dimz // 2. - 0.5)

            spec_emissionline = orb.utils.spectrum.sincgauss1d(
                np.arange(self.dimz),
                0, snr, channelr,
                sinc_fwhmr,
                sinc_sigmar)

            # create emission line source in 3d
            src3d += spec_emissionline
            src3d *= src2d

        # add noise +  random background
        out_list = list()
        for isample in range(samples):
            isrc3d = np.copy(src3d)

            # if has_source:
                # add source noise
                # source_noise = np.sqrt(src2d)
                # isrc3d += (np.random.standard_normal(src3d.shape) * source_noise)

            # add random background
            # X, Y = np.mgrid[:self.dimx:1., :self.dimy:1.]
            # X /= float(self.dimx)
            # Y /= float(self.dimy)

            # sky2d = (1 + X * np.sin(np.random.uniform(-self.TIPTILT, self.TIPTILT))
            #          + Y * np.sin(np.random.uniform(-self.TIPTILT,self.TIPTILT))).reshape(
            #          (self.dimx, self.dimy, 1))
            # sky2d -= np.median(sky2d)

            isky3d = self.background.get_random_box() #+ sky2d
            isky3d /= np.std(isky3d)
            isrc3d += isky3d

            # normalization
            # no need since with a snr of 0 the output has zero mean and unit variance

            out_list.append({'data': isrc3d.astype(np.float32),
                             'snr': np.array(snr).astype(np.float32),
                             'label': np.array(has_source).astype(np.int32)})
        if samples == 1:
            return out_list[0]
        else:
            return out_list

    def generate_graph(self):

        def add_dropout_layer(input_layer, dropout_rate):
            with tf.name_scope('dropout_layer'):
                return tf.layers.dropout(
                    input_layer, rate=dropout_rate,
                    training=self.is_training, name='dropout')

        
        def add_convolutional_layer(input_layer,
                                    filters, kernel_size,
                                    padding='SAME',
                                    max_pool=True,
                                    flatten=False,
                                    conv_stride=1):

            logging.debug('shape before entering convolution layer: {}'.format(input_layer.shape))

            POOL_STRIDE = 2

            with tf.name_scope('convolutional_layer'):
                input_layer = tf.layers.conv3d(input_layer, filters=filters,
                                         kernel_size=kernel_size,
                                         strides=conv_stride, padding=padding,
                                         activation=tf.nn.relu)

                if max_pool:
                    input_layer = tf.nn.max_pool3d(
                        input_layer,
                        ksize=[1, POOL_STRIDE, POOL_STRIDE, POOL_STRIDE, 1],
                        strides=[1, POOL_STRIDE, POOL_STRIDE, POOL_STRIDE, 1],
                        padding='VALID')

                if flatten:
                    logging.debug('convolutional layer before flattening shape: {}'.format(input_layer.shape))
                    input_layer = tf.reshape(
                        input_layer,
                        shape=[-1, orcs.utils.get_layer_size(input_layer)])


            logging.debug('convolutional layer output shape: {}'.format(input_layer.shape))
            return input_layer

        tf.reset_default_graph()

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32,
                shape=(None, np.multiply.reduce(self.shape)),
                name='X')
            pool3= tf.reshape(
                self.X,
                shape=[-1, self.dimx, self.dimy, self.dimz, 1])
            self.y = tf.placeholder(tf.int32, shape=(None), name='y')
            self.is_training = tf.placeholder(
                tf.bool, shape=(), name='is_training')
        
        pool3 = add_dropout_layer(pool3, self.INPUT_DROPOUT_RATE)

        pool3 = add_convolutional_layer(
            pool3,
            self.FILTERS, self.KERNEL_SIZE, max_pool=True, conv_stride=1,
            flatten=True)
        
        # pool3 = add_convolutional_layer(
        #     pool3, 
        #     self.FILTERS * 2, self.KERNEL_SIZE, max_pool=True, conv_stride=1,
        #     flatten=True)
        
        # pool3 = add_convolutional_layer(
        #     pool3,
        #     self.FILTERS * 4, self.KERNEL_SIZE, max_pool=True, conv_stride=1,
        #     flatten=True)
        
        with tf.name_scope("fc4"):
            fc4 = tf.layers.dense(
                pool3, pool3.shape[-1],
                activation= tf.nn.relu, name="fc4")
            fc4 = add_dropout_layer(fc4, self.OUTPUT_DROPOUT_RATE)

        with tf.name_scope("output"):
            logits = tf.layers.dense(fc4, 2, name="output")
            self.proba = tf.nn.softmax(logits, name="proba")

        with tf.name_scope("train"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.y)
            self.loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.LEARNING_RATE,
                epsilon=self.EPSILON,
                beta1=self.BETA1,
                beta2=self.BETA2)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


