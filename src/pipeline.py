from functools import partial

import numpy as np
import tensorflow as tf

from bebec2024.utils import he_to_freq


class CharacterizationPipeline:

    def __init__(self,
                strength_label='source_strength_estimated',
                dim=2,
                ref_mic_index=63,
                preprocess=True,
                neig=None,
                shift_loc=True, # either True or a float
                norm_loc='ap', # either 'ap' or a float or False/None
                freq_enc=False,
                freq_enc_type='he', # or he
                ):
        self.strength_label = strength_label
        self.dim = dim
        self.ref_mic_index = ref_mic_index
        self.preprocess = preprocess
        self.neig = 0 if neig is None else neig
        self.shift_loc = shift_loc
        self.norm_loc = norm_loc
        self.freq_enc = freq_enc
        self.freq_enc_type = freq_enc_type

    def transform_csm(self, data):
        data['csm']  = data['csm']/data['csm'][:,self.ref_mic_index,self.ref_mic_index][:,tf.newaxis,tf.newaxis]
        return data

    def recover_csm(self, csm):
        csm = csm * csm[:,self.ref_mic_index,self.ref_mic_index][:,tf.newaxis,tf.newaxis]
        return csm

    def transform_loc(self, aperture, data):
        if self.norm_loc:
            if isinstance(self.norm_loc, float):
                data['loc'] = data['loc'][:self.dim]/self.norm_loc
            else:
                data['loc'] = data['loc'][:self.dim]/aperture
        if self.shift_loc:
            if isinstance(self.shift_loc, float):
                data['loc'] = data['loc'] + self.shift_loc
            else:
                data['loc'] = data['loc'] + 0.5
        nf = tf.shape(data['f'])[0]
        data['loc'] = tf.repeat(data['loc'][tf.newaxis], repeats=nf, axis=0)
        return data

    def recover_loc(self, loc, aperture):
        if self.shift_loc:
            if isinstance(self.shift_loc, float):
                loc = loc - self.shift_loc
            else:
                loc = loc - 0.5
        if self.norm_loc:
            if isinstance(self.norm_loc, float):
                loc = loc * self.norm_loc
            else:
                loc = loc * aperture
        return loc

    def transform_strength(self, data):
        a = data[self.strength_label]
        data[self.strength_label] = a / tf.reduce_sum(a, axis=-1)[:,tf.newaxis]
        return data

    def get_lambda_v(self, data):
        csm = data['csm']
        evls, evecs = tf.linalg.eigh(csm)
        lambda_v = evecs[...,-self.neig:]*evls[:,tf.newaxis,-self.neig:]
        lambda_v = tf.stack([tf.math.real(lambda_v),tf.math.imag(lambda_v)],axis=3)
        lambda_v = tf.transpose(lambda_v,[0,2,1,3])
        input_shape = tf.shape(lambda_v)
        lambda_v = tf.reshape(lambda_v, [-1, input_shape[1],input_shape[2]*input_shape[3]])
        data['csm'] = lambda_v
        return data

    def add_freq_enc(self, freq_enc, features, labels):
        return ((features, freq_enc), labels)

    def get_features_and_labels(self, data):
        features = data['csm']
        labels = (data[self.strength_label], data['loc'])
        return (features, labels)

    def get_pipeline(self, dataset, batchsize, shuffle, shuffle_buffer_size, seed, **kwargs):
        num_mics = dataset.config.mics.num_mics
        max_nsources = dataset.config.max_nsources
        ap = dataset.config.mics.aperture
        c = dataset.config.env.c
        if self.neig == 0:
            neig = num_mics
        else:
            neig = self.neig

        # build pipeline
        tf_dataset = dataset.get_tf_dataset(**kwargs)
        tf_dataset = tf_dataset.map(self.transform_csm)
        tf_dataset = tf_dataset.map(partial(self.transform_loc, ap))
        tf_dataset = tf_dataset.map(self.transform_strength)
        if self.preprocess:
            tf_dataset = tf_dataset.map(self.get_lambda_v)
        tf_dataset = tf_dataset.map(self.get_features_and_labels)

        if self.freq_enc:
            freqs = kwargs['f']
            if self.freq_enc_type == 'ind':
                freq_enc = tf.range(start=0, limit=len(freqs), delta=1)
            elif self.freq_enc_type == 'he':
                freq_enc = tf.constant([ap*fftc/c for fftc in freqs], dtype=tf.float32)
            tf_dataset = tf_dataset.map(partial(self.add_freq_enc, freq_enc))

        # pad, shuffle and batch
        tf_dataset = tf_dataset.unbatch()
        if shuffle:
            tf_dataset = tf_dataset.shuffle(shuffle_buffer_size, seed=seed)

        if self.freq_enc and self.preprocess:
            tf_dataset = tf_dataset.padded_batch(
            batchsize, padded_shapes=(((neig, num_mics*2),()), ((max_nsources), (self.dim, max_nsources))),
            padding_values=((tf.constant(0,dtype=tf.float32), 0.), (0.,0.)), drop_remainder=True
            )
        elif not self.freq_enc and self.preprocess:
            tf_dataset = tf_dataset.padded_batch(
            batchsize, padded_shapes=((neig, num_mics*2), ((max_nsources), (self.dim, max_nsources))),
            padding_values=(tf.constant(0,dtype=tf.float32), (0.,0.)), drop_remainder=True
            )
        elif self.freq_enc and not self.preprocess:
            tf_dataset = tf_dataset.padded_batch(
            batchsize, padded_shapes=(((num_mics, num_mics),()), ((max_nsources), (self.dim, max_nsources))),
            padding_values=((tf.constant(0,dtype=tf.complex64), 0.), (0.,0.)), drop_remainder=True
            )
        elif not self.freq_enc and not self.preprocess:
            tf_dataset = tf_dataset.padded_batch(
            batchsize, padded_shapes=((num_mics, num_mics), ((max_nsources), (self.dim, max_nsources))),
            padding_values=(tf.constant(0,dtype=tf.complex64), (0.,0.)), drop_remainder=True
            )
        return tf_dataset

    @staticmethod
    def filter_kwargs(dataset, **kwargs):
        """Provide an AcouPipe ready kwargs dict that can be passed to a <dataset>.generate(**kwargs)_method."""
        if kwargs.get('f') is None and 'he_min' in kwargs and 'he_max' in kwargs:
            he_min = kwargs.pop('he_min')
            he_max = kwargs.pop('he_max')
            kwargs['f'] = he_to_freq(dataset, he_min=he_min, he_max=he_max)
        elif 'f' not in kwargs:
            raise ValueError('f is not defined in kwargs, and he_min and/or he_max are not defined either')
        else:
            kwargs.pop('he_min', None)
            kwargs.pop('he_max', None)
        return kwargs

    def __call__(self, dataset, batchsize, shuffle=False, shuffle_buffer_size=1, cache=False, prefetch_size=None,
            seed=1, **kwargs):
        self.seed = seed
        kwargs = self.filter_kwargs(dataset, **kwargs)

        if 'features' in kwargs and kwargs['features'] is not None:
            raise ValueError('features is already defined in kwargs')
        else:
            kwargs['features'] = ['csm','loc',self.strength_label,'f']
        print(kwargs)
        dataset = self.get_pipeline(dataset, batchsize, shuffle, shuffle_buffer_size, seed, **kwargs)
        if cache:
            dataset = dataset.cache()
        if prefetch_size is not None:
            dataset = dataset.prefetch(prefetch_size)
        return dataset
