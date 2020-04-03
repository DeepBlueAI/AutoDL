#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:12
# @Author:  Mecthew

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.layers import (Activation, Flatten, Conv2D,Input,LeakyReLU,MaxPooling1D,concatenate,GlobalMaxPooling1D,
                                            MaxPooling2D, BatchNormalization)
from tensorflow.python.keras.layers import (Conv1D, Dense, Dropout, MaxPool1D)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model as TFModel
import abc
class ModelBase(object):
    @abc.abstractmethod
    def init_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess_data(self, x):
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, test_x):
        pass
def ohe2cat(label):
    return np.argmax(label, axis=1)

# from CONSTANT import IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
# from data_process import ohe2cat, extract_mfcc, get_max_length, pad_seq, extract_mfcc_parallel
# from models.my_classifier import Classifier
# from tools import log
# from tools import timeit


# Input of CNN1D is speech raw data， shape of each sample is
# (sample_rate*default_duration, 1)
class CnnModel1D(ModelBase):
    def __init__(self):
        self.max_length = None


        self._model = None
        self.is_init = False
    def init_model(self,config):

        input_shape = config['max_len']
        num_classes = config['num_classes']

        inputs = Input(shape=(input_shape,96))
        x = inputs
        cnn1 = Conv1D(50, kernel_size=1,
                      strides=1, padding='same',
                      kernel_initializer='he_normal')(x)
        cnn1 = BatchNormalization(axis=-1)(cnn1)
        cnn1 = LeakyReLU()(cnn1)
        cnn1 = GlobalMaxPooling1D()(cnn1)  # CNN_Dynamic_MaxPooling(cnn1,50,2,2)

        cnn2 = Conv1D(50, kernel_size=3,
                      strides=1, padding='same',
                      kernel_initializer='he_normal')(x)
        cnn2 = BatchNormalization(axis=-1)(cnn2)
        cnn2 = LeakyReLU()(cnn2)
        cnn2 = GlobalMaxPooling1D()(cnn2)

        cnn3 = Conv1D(50, kernel_size=5,
                      strides=1, padding='same',
                      kernel_initializer='he_normal')(x)
        cnn3 = BatchNormalization(axis=-1)(cnn3)
        cnn3 = LeakyReLU()(cnn3)
        cnn3 = GlobalMaxPooling1D()(cnn3)
        x = concatenate([cnn1, cnn2, cnn3], axis=-1)

        x = Dense(units=num_classes, activation='softmax')(x)
        model = TFModel(inputs=inputs, outputs=x)
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=['acc'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, **kwargs):
        val_x, val_y = validation_data_fit
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)]
        epochs = 10
        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)
    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)


