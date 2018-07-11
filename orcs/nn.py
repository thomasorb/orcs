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

import orcs.core
import orcs.utils
import orb.utils.io
import orb.utils.spectrum
import orb.cutils
import orb.core
import orb.utils.validate
import orb.utils.image

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class NNWorker(orb.core.Tools):

    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1

    def __init__(self, shape, data_prefix=None):
        if data_prefix is None:
            data_prefix = self.__class__.__name__ + '.'
        elif not isinstance(data_prefix, str):
            raise TypeError('data_prefix must be a string')

        self.data_prefix = orcs.core.DataFiles().get_path(data_prefix)

        orb.utils.validate.is_iterable(shape, object_name='shape')
        self.shape = shape

        self.dimx = self.shape[0]
        if len(shape) > 1:
            self.dimy = self.shape[1]
        if len(shape) > 2:
            self.dimz = self.shape[2]
        if self.TRAIN_RATIO + self.VALID_RATIO + self.TEST_RATIO != 1:
            raise ValueError('TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1')

        self.dataset = None

    def get_session(self):
        return tf.Session(config=tf.ConfigProto(log_device_placement=True))

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

    def train(self, n_epochs=4, batch_size=500):

        def shuffle_batch(X, y, batch_size):
            rnd_idx = np.random.permutation(len(X))
            n_batches = len(X) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                yield X_batch, y_batch

        self.load_dataset()
        self.generate_graph()

        with self.get_session() as sess:
            self.init.run()

            for epoch in range(n_epochs):
                for X_batch, y_batch in shuffle_batch(
                    self.dataset['data_train'],
                    self.dataset['label_train'],
                    batch_size):
                    sess.run(self.training_op,
                             feed_dict={self.X: X_batch, self.y: y_batch})
                acc_batch = self.accuracy.eval(
                    feed_dict={self.X: X_batch, self.y: y_batch})
                acc_val = self.accuracy.eval(
                    feed_dict={self.X: self.dataset['data_valid'],
                               self.y: self.dataset['label_valid']})
                logging.info('epoch {}: batch accuracy: {}, val accuracy: {}'.format(
                    epoch, acc_batch, acc_val))

                self.saver.save(sess, self.get_model_path())
                logging.info('model saved as {}'.format(self.get_model_path()))


    def test(self):
        self.load_dataset()
        predictions, probas = self.run_on_sample(self.dataset['data_test'])

        diff = (predictions - self.dataset['label_test']) != 0
        logging.info('number of predictions: {}'.format(diff.size))
        logging.info('number of errors: {}'.format(np.sum(diff)))

        return predictions, probas

    def run_on_sample(self, sample):
        self.generate_graph()
        with self.get_session() as sess:
            self.saver.restore(sess, self.get_model_path())
            probas = self.proba.eval(
                feed_dict={self.X: sample})
            predictions = tf.argmax(probas, axis=1).eval()
        return predictions, probas


    def run(self):
        pass

class SourceDetector3d(NNWorker):

    DIMX = 8  # x size of the box
    DIMY = 8  # y size of the box
    DIMZ = 4  # number of channels in the box
    SOURCE_FWHM = [2.5, 3.5]  # source FWHM
    SINC_WIDTH = [0.9,1.1]  # SINC FWHM = 1.20671 * WIDTH
    TIPTILT = np.pi / 8
    SNR = (0.1, 30)  # min max SNR - log uniform distribution

    def __init__(self):

        NNWorker.__init__(self, (self.DIMX, self.DIMY, self.DIMZ))

    def simulate(self, snr=None, samples=1):

        dxr = self.dimx/2. - 1 + np.random.uniform()
        dyr = self.dimy/2. - 1 + np.random.uniform()

        src3d = np.zeros(self.shape, dtype=np.float32)
        #src3d_continuum = np.zeros(self.shape, dtype=np.float32)
        src3d_sky = np.zeros(self.shape, dtype=np.float32)

        if snr is None:
            has_source = np.random.randint(0, 2)
            if has_source: # log uniform distribution of the SNR
                snr = 10**(np.random.uniform(
                    np.log10(self.SNR[0]),
                    np.log10(self.SNR[1])))
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
            channelr = ((np.random.uniform() - 0.5) + self.dimz / 2. - 0.5)
            spec_emissionline = orb.utils.spectrum.sinc1d(
                np.arange(self.dimz),
                0, 1, channelr,
                sinc_widthr)

            # create emission line source in 3d
            src3d += spec_emissionline
            src3d *= src2d

        # compute noise level
        if has_source:
            noise = 1. / snr
        else: noise = 1.

        # create continuum source in 3d
        #spec_continuum = np.random.uniform(0.9, 1.1, self.dimz)
        #src3d_continuum += spec_continuum
        #src3d_continuum *= src2d_continuum

        # create sky spectrum
        spec_sky = noise * 3
        X, Y = np.mgrid[:self.dimx:1., :self.dimy:1.]
        X /= float(self.dimx)
        Y /= float(self.dimy)

        sky2d = (1 + X * np.sin(np.random.uniform(-self.TIPTILT, self.TIPTILT))
                 + Y * np.sin(np.random.uniform(-self.TIPTILT,self.TIPTILT)))
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
            isrc3d /= np.max(isrc3d)

            out_list.append({'data': isrc3d.astype(np.float32),
                             'snr': np.array(snr).astype(np.float32),
                             'label': np.array(has_source).astype(np.int32)})
        if samples == 1:
            return out_list[0]
        else:
            return out_list

    def generate_graph(self):

        CONV1_FILTERS = 4
        CONV1_KERNEL = 3
        CONV1_STRIDE = 2
        CONV1_PAD = "SAME"

        CONV2_FILTERS = 8
        CONV2_KERNEL = 3
        CONV2_STRIDE = 2
        CONV2_PAD = "SAME"

        POOL3_FILTERS = 8
        POOL3_STRIDE = 2
        POOL3_KERNEL = 2

        FULLY_CONNECTED_LAYER_SIZE = 8




        def add_convolutional_layer(input_layer, input_shape,
                                    filters, kernel_size=3,
                                    padding='SAME', flatten=False):

            total_stride = 2
            if len(input_shape) != 3:
                raise TypeError('input shape must be a tuple (dimx, dimy, dimz)')
            output_shape = [input_shape[0]//total_stride,
                            input_shape[1]//total_stride,
                            filters]

            with tf.name_scope('convolution_layer'):
                input_layer = tf.layers.conv2d(input_layer, filters=filters,
                                         kernel_size=kernel_size,
                                         strides=1, padding=padding,
                                         activation=None)

                input_layer = tf.layers.conv2d(input_layer, filters=filters,
                                         kernel_size=kernel_size,
                                         strides=1, padding=padding,
                                         activation=tf.nn.relu)

                input_layer = tf.nn.max_pool(input_layer,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding="VALID")
                if flatten:
                    input_layer = tf.reshape(
                        input_layer,
                        shape=[-1, np.multiply.reduce(output_shape)])
                    output_shape = [np.multiply.reduce(output_shape)]

                return input_layer, output_shape



        tf.reset_default_graph()

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32,
                shape=(None, np.multiply.reduce(self.shape)),
                name='X')
            self.X_reshaped = tf.reshape(
                self.X,
                shape=[-1, self.dimx, self.dimy, self.dimz])
            self.y = tf.placeholder(tf.int32, shape=(None), name='y')

        pool3, pool3_shape = add_convolutional_layer(
            self.X_reshaped, [self.dimy, self.dimx, self.dimz], 16)

        pool3, pool3_shape = add_convolutional_layer(
            pool3, pool3_shape, 16)

        pool3, pool3_shape = add_convolutional_layer(
            pool3, pool3_shape, 16, flatten=True)

        with tf.name_scope("fc4"):

            fc4 = tf.layers.dense(
                pool3, FULLY_CONNECTED_LAYER_SIZE,
                activation=tf.nn.relu, name="fc4")

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


    def run(self, cube, start=None, stop=None):
        self.generate_graph()

        streamer = cube.get_streamer(
            bsize=[self.dimx, self.dimy],
            start=start,
            stop=stop,
            strides=[1, 1])

        with self.get_session() as sess:
            self.saver.restore(sess, self.get_model_path())

            for iquad in streamer:
                iquad[np.nonzero(np.isnan(iquad))] = 0
                dataset = list()
                for iz in range(0, iquad.shape[2] - self.dimz):
                    dataset.append(iquad[:, :, iz:iz + self.dimz].flatten())

                dataset_normed = dataset / np.max(np.array(dataset))
                probas = self.proba.eval(
                    feed_dict={self.X: dataset_normed})
                predictions = tf.argmax(probas, axis=1).eval()
                print predictions

class HDFCube(orcs.core.HDFCube):

    def get_streamer(self, bsize=[16, 16], start=None, stop=None, strides=[1,1]):

        slices = orcs.utils.image_streamer(
            self.dimx, self.dimy, bsize, start=start,
            stop=stop, strides=strides)

        for islice in slices:
            yield self[islice[0], islice[1], :]
