#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/27 10:12
# @Author:  Mecthew

import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (SpatialDropout1D, Input, GlobalMaxPool1D,Bidirectional,GlobalAvgPool1D,concatenate,CuDNNGRU,
                                            Dense, Dropout, CuDNNLSTM, Activation, Lambda, Flatten,Input, Dense, Dropout, Convolution2D,
                                            MaxPooling2D, ELU, Reshape, CuDNNGRU)
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model as TFModel

from CONSTANT import MAX_FRAME_NUM, IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import (extract_mfcc_parallel)
from data_process import ohe2cat, get_max_length, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tools import log


class LstmAttention(Classifier):
    def __init__(self):
        # clear_session()
        log("new LSTM")
        self.max_length = None
        self._model = None
        self.is_init = False
        self.epoch_cnt = 0
        self.mfcc_mean, self.mfcc_std = None, None
        self.mel_mean, self.mel_std = None, None
        self.cent_mean, self.cent_std = None, None
        self.stft_mean, self.stft_std = None, None

    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x_mfcc = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x_mfcc)
            self.max_length = min(MAX_FRAME_NUM, self.max_length)
        x_mfcc = pad_seq(x_mfcc, pad_len=self.max_length)

        return x_mfcc

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        freq_axis = 2
        channel_axis = 3
        channel_size = 128
        min_size = min(input_shape[:2])
        melgram_input = Input(shape=input_shape)
        # x = ZeroPadding2D(padding=(0, 37))(melgram_input)
        # x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

        x = Reshape((input_shape[0], input_shape[1], 1))(melgram_input)
        # Conv block 1
        x = Convolution2D(64, 3, 1, padding='same', name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='bn1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = Dropout(0.1, name='dropout1')(x)

        # Conv block 2
        x = Convolution2D(channel_size, 3, 1, padding='same', name='conv2')(x)
        x = BatchNormalization(axis=channel_axis, name='bn2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
        x = Dropout(0.1, name='dropout2')(x)

        # Conv block 3
        x = Convolution2D(channel_size, 3, 1, padding='same', name='conv3')(x)
        x = BatchNormalization(axis=channel_axis, name='bn3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, min_size/6), strides=(3, min_size/6), name='pool3')(x)
        x = Dropout(0.1, name='dropout3')(x)

        # if min_size // 24 >= 4:
        #     # Conv block 4
        #     x = Convolution2D(
        #         channel_size,
        #         3,
        #         1,
        #         padding='same',
        #         name='conv4')(x)
        #     x = BatchNormalization(axis=channel_axis, name='bn4')(x)
        #     x = ELU()(x)
        #     x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
        #     x = Dropout(0.1, name='dropout4')(x)

        x = Reshape((-1, channel_size))(x)
        avg = GlobalAvgPool1D()(x)
        max = GlobalMaxPool1D()(x)
        x = concatenate([avg,max],axis=-1)
        # x = Dense(max(int(num_classes*1.5), 128), activation='relu', name='dense1')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', name='output')(x)

        model = TFModel(inputs=melgram_input, outputs=outputs)


        optimizer = optimizers.Nadam(
            lr=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            schedule_decay=0.004)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, round_num, cur_model_loop,**kwargs):
        val_x, val_y = validation_data_fit

        # if train_loop_num == 1:
        #     patience = 2
        #     epochs = 3
        # elif train_loop_num == 2:
        #     patience = 3
        #     epochs = 10
        # elif train_loop_num < 10:
        #     patience = 4
        #     epochs = 16
        # elif train_loop_num < 15:
        #     patience = 4
        #     epochs = 24
        # else:
        #     patience = 8
        #     epochs = 32

        patience = 2
        # epochs = self.epoch_cnt + 3
        epochs = 10
        batch_size = 32
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)]
        # if round_num>0 and cur_model_loop>=10:
        #     batch_size = 16
        #     callbacks = [
        #         tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss',
        #             patience=patience),
        #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0005, verbose=2)
        #     ]


        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        # validation_split=0.2,
                        verbose=1,  # Logs once per epoch.
                        batch_size=batch_size,
                        shuffle=True,
                        # initial_epoch=self.epoch_cnt,
                        # use_multiprocessing=True
                        )
        self.epoch_cnt += 3

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
