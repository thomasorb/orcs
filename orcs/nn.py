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

class NNWorker(orb.core.Tools):

    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1
    BATCH_SIZE = 1000

    def __init__(self, shape, data_prefix=None):
        if data_prefix is None:
            data_prefix = self.__class__.__name__ + '.'
        elif not isinstance(data_prefix, str):
            raise TypeError('data_prefix must be a string')

        self.data_prefix = orcs.core.DataFiles().get_path(data_prefix)

        orb.utils.validate.is_iterable(shape, object_name='shape')
        self.shape = shape

        if int(self.shape[0])&1: raise ValueError('shape[0] must be even')
        self.dimx = int(self.shape[0])
        if len(shape) > 1:
            if int(self.shape[1])&1: raise ValueError('shape[1] must be even')
            self.dimy = int(self.shape[1])
        if len(shape) > 2:
            if int(self.shape[2])&1: raise ValueError('shape[2] must be even')
            self.dimz = int(self.shape[2])
        if self.TRAIN_RATIO + self.VALID_RATIO + self.TEST_RATIO != 1:
            raise ValueError('TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1')

        self.dataset = None
        self.is_training = True

    def get_session(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        return self.session

    def init_session(self, is_training=False):
        if is_training:
            self.load_dataset()
            self.is_training = True
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
            valid_slice = slice(
                train_slice.stop, train_slice.stop + int(size * self.VALID_RATIO))
            test_slice = slice(valid_slice.stop, size)

            logging.info('dataset size: {}'.format(size))
            for ikey in inf.keys():
                idat = inf[ikey][:]
                if np.any(np.isnan(idat)): raise Exception('nan in data')
                if np.any(np.isinf(idat)): raise Exception('inf in data')
                train_key = ikey + '_train'
                valid_key = ikey + '_valid'
                test_key = ikey + '_test'

                self.dataset[train_key] = np.squeeze(idat[train_slice,:])
                self.dataset[valid_key] = np.squeeze(idat[valid_slice,:])
                self.dataset[test_key] = np.squeeze(idat[test_slice,:])

                logging.info('{} shape: {}'.format(
                    train_key, self.dataset[train_key].shape))
                logging.info('{} shape: {}'.format(
                    valid_key, self.dataset[valid_key].shape))
                logging.info('{} shape: {}'.format(
                    test_key, self.dataset[test_key].shape))

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
                        rotated_sim_dict['data'] = np.flip(sim_dict['data'], 0)
                        rotated_sim_dict['data'] = np.flip(sim_dict['data'], 1)

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

    def train(self, n_epochs=4, sorting_dict=None):

        def shuffle_batch(dataset, batch_size, sorting_dict=None):
            """
            :param sorting_dict: must be a dictionary giving for each label its
              sorting key (e.g. {1:'snr'} to sort label 1 by snr (), if a 'snr'
              is provided in the dataset).
            """
            if sorting_dict is not None:
                if not isinstance(sorting_dict, dict):
                    raise TypeError('sorting_dict must be a dict')

            rnd_idx = np.random.permutation(len(self.dataset['data_train']))

            if sorting_dict is not None:
                for ilabel in sorting_dict:
                    # get indexes corresponding to the label in the permuted dataset
                    label_idx = np.nonzero(
                        self.dataset['label_train'][rnd_idx] == ilabel)[0]
                    # sort indexes corresponding to the label against the set
                    # specified in sorting_dict (high values first)
                    sorted_label_idx = label_idx[np.argsort(
                        self.dataset[sorting_dict[ilabel] + '_train'][rnd_idx][label_idx])[::-1]]
                    # replace the unsorted indexes corresponding to the label with
                    # the sorted ones
                    rnd_idx[label_idx] = rnd_idx[sorted_label_idx]

            n_batches = len(self.dataset['data_train']) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                if sorting_dict is not None:
                    # each batch is shuffled again because some labels have been
                    # sorted in the batch
                    np.random.shuffle(batch_idx)

                X_batch = self.dataset['data_train'][batch_idx]
                y_batch = self.dataset['label_train'][batch_idx]
                yield X_batch, y_batch


        with self.init_session(is_training=True):

            for epoch in range(n_epochs):
                stime = time.time()

                progress = orb.core.ProgressBar(
                    len(self.dataset['data_train']) // self.BATCH_SIZE)
                iloop = 1
                for X_batch, y_batch in shuffle_batch(
                    self.dataset,
                    self.BATCH_SIZE, sorting_dict=sorting_dict):
                    progress.update(iloop, info='training')
                    self.session.run(self.training_op,
                             feed_dict={self.X: X_batch, self.y: y_batch})
                    iloop += 1
                progress.end()

                acc_batch = self.accuracy.eval(
                    feed_dict={self.X: X_batch, self.y: y_batch})

                acc_val_list = list()

                for X_batch, y_batch in shuffle_batch(
                    self.dataset,
                    self.BATCH_SIZE, sorting_dict=sorting_dict):
                    acc_val_list.append(self.accuracy.eval(
                        feed_dict={self.X: X_batch, self.y: y_batch}))
                logging.info('training time: {} epoch {}: batch accuracy: {}, val accuracy: {} [{}]'.format(
                    time.time() - stime, epoch, acc_batch,
                    np.median(acc_val_list), np.std(acc_val_list)))

                self.save_graph()
                logging.info('model saved as {}'.format(self.get_model_path()))


    def test(self):
        self.load_dataset()
        predictions, probas = self.run_on_sample(self.dataset['data_test'])

        diff = (predictions - self.dataset['label_test']) != 0
        logging.info('number of predictions: {}'.format(diff.size))
        logging.info('number of errors: {}'.format(np.sum(diff)))

        return predictions, probas

    def run_on_sample(self, sample):
        sample = np.array(sample)
        with self.init_session():

            probas_list = list()
            predictions_list = list()
            for ibatch in range(0, sample.shape[0], self.BATCH_SIZE):
                X_batch = sample[ibatch:ibatch+self.BATCH_SIZE, ...]
                probas_list.append(np.array(self.proba.eval(
                    feed_dict={self.X: X_batch})))
                predictions_list.append(
                    np.array(tf.argmax(probas_list[-1], axis=1).eval()))

            predictions = np.concatenate(predictions_list)
            probas = np.concatenate(probas_list)

        return predictions, probas

    def _run_on_cube(self, cube, zlimits=None, slices=None):
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
            output['proba_highest'] = list()
            output['proba_arghighest'] = list()


            # run detection
            progress = orb.core.ProgressBar(slices_nb)
            iloop = 0
            for islice in slices:
                if not iloop%100:
                    progress.update(iloop, 'running on slice {}/{}'.format(iloop, slices_nb))

                if zlimits is None:
                    idata_box = np.copy(cube[islice[0], islice[1], :])
                else:
                    idata_box = np.copy(cube[islice[0], islice[1], zlimits[0]:zlimits[1]])

                idata_box[np.nonzero(np.isnan(idata_box))] = 0.
                imedian = np.median(idata_box, axis=[0,1])
                #imax = np.max(imedian)
                imax = np.percentile(idata_box, 99.95)

                idata_box -= imedian
                idata_box /= imax

                boxes = list()
                for ibox in box_streamer(idata_box, self.dimz):
                    boxes.append(ibox)
                boxes = np.array(boxes)

                probas_list = list()
                for ibatch in range(0, boxes.shape[0], self.BATCH_SIZE):

                    X_batch = boxes[ibatch:ibatch+self.BATCH_SIZE, ...]
                    probas_list.append(np.array(self.proba.eval(
                        feed_dict={self.X: X_batch})))

                ## probas must be outputed for each label !!
                iprobas = np.concatenate(probas_list)
                iprobas_sorted = np.argsort(iprobas, axis=0)
                output['proba_argmax'].append(iprobas_sorted[-1,:])
                output['proba_max'].append(
                    np.diag(iprobas[iprobas_sorted[-1,:]]))
                output['proba_median'].append(
                    np.diag(iprobas[iprobas_sorted[iprobas.shape[0]//2,:]]))
                output['proba_min'].append(
                    np.diag(iprobas[iprobas_sorted[0,:]]))
                output['proba_arghighest'].append(
                    iprobas_sorted[-HIGHEST_NB:,:])
                output['proba_highest'].append(
                    np.diagonal(iprobas[iprobas_sorted[-HIGHEST_NB:,:]],
                                axis1=2))

                iloop += 1
            progress.end()

        formatted_output = list()
        for iclass in range(self.nclasses):
            formatted_output.append(dict())
            for ikey in output:
                output[ikey] = np.array(output[ikey])
                formatted_output[iclass][ikey] = list(output[ikey][...,iclass])
        return formatted_output


    def run_on_cube_targets(self, cube, targets):

        cube._silent_load = True
        zlimits = np.array(cube.get_filter_range_pix()).astype(int)

        targets = np.array(targets)
        if targets.shape[1] != 2:
            raise TypeError('badly formatted target list. Must have shape (n, 2) but has shape {}'.format(targets.shape))

        # check targets
        xmin = orb.utils.validate.index(
            targets[:,0] - self.dimx//2, 0, cube.dimx - self.dimx, clip=False)
        xmax = orb.utils.validate.index(
            targets[:,0] + self.dimx//2, 0, cube.dimx, clip=False)
        ymin = orb.utils.validate.index(
            targets[:,1] - self.dimy//2, 0, cube.dimy - self.dimy, clip=False)
        ymax = orb.utils.validate.index(
            targets[:,1] + self.dimy//2, 0, cube.dimy, clip=False)

        slices = list()
        for i in range(targets.shape[0]):
            slices.append([slice(xmin[i], xmax[i]), slice(ymin[i], ymax[i])])

        return self._run_on_cube(cube, zlimits=zlimits, slices=slices)


    def run_on_cube(self, cube, start, stop):

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
        if data_surface > 0.05 * cube.dimx * cube.dimy:
            raise NotImplementedError()

        idata_slice = cube.get_data(data_xrange[0], data_xrange[-1],
                                    data_yrange[0], data_yrange[-1],
                                    zlimits[0],
                                    zlimits[1] - self.dimz,
                                    silent=False)

        output = self._run_on_cube(idata_slice, zlimits=None)

        timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        for iclass in range(len(output)):
            for ikey in output[iclass]:
                output[iclass][ikey] = np.reshape(
                    output[iclass][ikey],
                    (stop[0] - start[0], stop[1] - start[1], -1))
                orb.utils.io.write_fits(
                    './{}/outmap.{}.{}.fits'.format(timestamp, iclass, ikey),
                    output[iclass][ikey], overwrite=True)






        #

        return output

class SourceDetector3d(NNWorker):

    DIMX = 8  # x size of the box
    DIMY = 8  # y size of the box
    DIMZ = 8  # number of channels in the box
    SOURCE_FWHM = [2.5, 3.5]  # source FWHM
    SINC_WIDTH = [0.9,1.1]  # SINC FWHM = 1.20671 * WIDTH
    TIPTILT = np.pi / 8
    SNR = (2.5, 30)  # min max SNR (log uniform)

    def __init__(self):

        NNWorker.__init__(self, (self.DIMX, self.DIMY, self.DIMZ))

    def simulate(self, snr=None, samples=1):

        dxr = self.dimx/2. - 0.5 + np.random.uniform(-0.5, 0.5)
        dyr = self.dimy/2. - 0.5 + np.random.uniform(-0.5, 0.5)

        src3d = np.zeros(self.shape, dtype=np.float32)
        #src3d_continuum = np.zeros(self.shape, dtype=np.float32)
        src3d_sky = np.zeros(self.shape, dtype=np.float32)

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

        source_fwhmr = np.random.uniform(self.SOURCE_FWHM[0], self.SOURCE_FWHM[1])

        src2d_normalized = orb.cutils.gaussian_array2d(
            0, 1,
            dxr, dyr,
            source_fwhmr,
            self.dimx, self.dimy)
        src2d_normalized = src2d_normalized.reshape((self.dimx, self.dimy, 1))

        # emission line source
        if has_source:
            src2d = np.copy(src2d_normalized)
        else:
            src2d = 0.

        # continuum source
        #source_backgroundr = np.random.uniform(0., 0.1) *  self.SKY_BACKGROUND
        #src2d_continuum = np.copy(src2d_normalized) * source_backgroundr

        # emission line spectrum
        if has_source:
            sinc_widthr = np.random.uniform(self.SINC_WIDTH[0], self.SINC_WIDTH[1])
            channelr = (np.random.uniform(-0.5, 0.5) + self.dimz / 2. - 0.5)
            spec_emissionline = orb.utils.spectrum.sinc1d(
                np.arange(self.dimz),
                0, snr, channelr,
                sinc_widthr)

            # create emission line source in 3d
            src3d += spec_emissionline
            src3d *= src2d

        # compute noise level
        if has_source:
            noise = 1.
        else: noise = 1.

        # create continuum source in 3d
        #spec_continuum = np.random.uniform(0.9, 1.1, self.dimz)
        #src3d_continuum += spec_continuum
        #src3d_continuum *= src2d_continuum

        # create sky background
        spec_sky = np.random.uniform(size=self.dimz) * 0.1#0
        X, Y = np.mgrid[:self.dimx:1., :self.dimy:1.]
        X /= float(self.dimx)
        Y /= float(self.dimy)

        sky2d = (1 + X * np.sin(np.random.uniform(-self.TIPTILT, self.TIPTILT))
                 + Y * np.sin(np.random.uniform(-self.TIPTILT,self.TIPTILT)))
        #sky2d -= np.mean(sky2d)
        sky2d = sky2d.reshape((self.dimx, self.dimy, 1))
        src3d_sky += spec_sky
        src3d_sky *= sky2d

        # merge all
        src3d += src3d_sky #+ src3d_continuum

        # add noise
        out_list = list()
        for isample in range(samples):
            isrc3d = np.copy(src3d)

            isrc3d += (np.random.standard_normal(src3d.shape) * noise)

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

        def add_convolutional_layer(input_layer, input_shape,
                                    filters, kernel_size=2,
                                    padding='SAME',
                                    max_pool=True,
                                    flatten=False,
                                    conv_stride=1):

            POOL_STRIDE = 2
            if max_pool:
                total_stride = POOL_STRIDE * conv_stride
            else:
                total_stride = 1 * conv_stride

            #print input_shape, total_stride

            if len(input_shape) != 4:
                raise TypeError('input shape must be a tuple (dimx, dimy, dimz, filters)')
            output_shape = [input_shape[0]//total_stride,
                            input_shape[1]//total_stride,
                            input_shape[2]//total_stride,
                            filters]

            with tf.name_scope('convolution_layer'):
                input_layer = tf.layers.conv3d(input_layer, filters=filters,
                                         kernel_size=kernel_size,
                                         strides=conv_stride, padding=padding,
                                         activation=tf.nn.relu)

                if max_pool:
                    input_layer = tf.nn.max_pool3d(input_layer,
                                           ksize=[1, POOL_STRIDE, POOL_STRIDE, POOL_STRIDE, 1],
                                           strides=[1, POOL_STRIDE, POOL_STRIDE, POOL_STRIDE, 1],
                                           padding=padding)
                if flatten:
                    input_layer = tf.reshape(
                        input_layer,
                        shape=[-1, np.multiply.reduce(output_shape)])
                    output_shape = [np.multiply.reduce(output_shape)]

                #print output_shape
                return input_layer, output_shape

        FILTERS = 8

        tf.reset_default_graph()

        #is_training = tf.placeholder(tf.bool, shape = (), name='is_training')


        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32,
                shape=(None, np.multiply.reduce(self.shape)),
                name='X')
            X_reshaped= tf.reshape(
                self.X,
                shape=[-1, self.dimx, self.dimy, self.dimz, 1])
            self.y = tf.placeholder(tf.int32, shape=(None), name='y')

        pool3, pool3_shape = add_convolutional_layer(
            X_reshaped, [self.dimy, self.dimx, self.dimz, 1],
            FILTERS, max_pool=True, conv_stride=1)

        # pool3, pool3_shape = add_convolutional_layer(
        #     pool3, pool3_shape,
        #     FILTERS * 2, max_pool=True, conv_stride=1)

        pool3, pool3_shape = add_convolutional_layer(
            pool3, pool3_shape,
            FILTERS * 2, max_pool=True, conv_stride=1,
            flatten=True)


        with tf.name_scope("fc4"):
            fc4 = tf.layers.dense(
                pool3, pool3_shape[0] // 2,
                activation= tf.nn.relu, name="fc4")

        with tf.name_scope("output"):
            logits = tf.layers.dense(fc4, 2, name="output")
            self.proba = tf.nn.softmax(logits, name="proba")

        with tf.name_scope("train"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.y)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()



            # COLUMNS_PER_SLICE = 20
            # slice_range = np.arange(start[0], stop[0], COLUMNS_PER_SLICE)
            # #yrange = np.arange(start[1] - self.dimx//2, stop[1] + self.dimx//2)
            # for islice in slice_range:
            #     column_slice = cube.get_data(
            #         ii - self.dimx//2, ii + self.dimx//2 + 1,
            #         start[1], stop[1] + 1,
            #         0, cube.dimz, silent=True)
            #         print data_slice.shape
            #     return


            #
            # slices = orcs.utils.image_streamer(
            #     self.dimx, self.dimy, bsize, start=start,
            #     stop=stop, strides=strides)
            #
            # for islice in slices:
            #     yield self[islice[0], islice[1], :]
            #
            # for iquad in streamer:
            #     iquad[np.nonzero(np.isnan(iquad))] = 0
            #     dataset = list()
            #     for iz in range(0, iquad.shape[2] - self.dimz):
            #         dataset.append(iquad[:, :, iz:iz + self.dimz].flatten())
            #
            #     dataset_normed = dataset / np.max(np.array(dataset))
            #     probas = self.proba.eval(
            #         feed_dict={self.X: dataset_normed})
            #     predictions = tf.argmax(probas, axis=1).eval()


# class HDFCube(orcs.core.HDFCube):
#
#     def get_streamer(self, bsize=[16, 16], start=None, stop=None, strides=[1,1]):
