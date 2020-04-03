#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/27 10:12
# @Author:  Mecthew
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (SpatialDropout1D, Input, GlobalMaxPool1D,Bidirectional,GlobalAvgPool1D,concatenate,CuDNNGRU,
                                            Dense, Dropout, CuDNNLSTM, Activation, Lambda, Flatten,Input, Dense, Dropout, Convolution2D,
                                            MaxPooling2D, ELU, Reshape, CuDNNGRU,Average)
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model as TFModel

from CONSTANT import IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import ohe2cat, extract_mfcc_parallel, get_max_length, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tools import log


class BilstmAttention(Classifier):
    def __init__(self):
        # clear_session()
        log('init BilstmAttention')
        self.max_length = None
        self._model = None
        self.is_init = False

    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x)
            self.max_length = min(800, self.max_length)
        x = pad_seq(x, pad_len=self.max_length)
        return x

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):

        freq_axis = 2
        channel_axis = 3
        channel_size = 128
        min_size = min(input_shape[:2])
        inputs = Input(shape=input_shape)
        # x = ZeroPadding2D(padding=(0, 37))(melgram_input)
        # x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

        x = Reshape((input_shape[0], input_shape[1], 1))(inputs)
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
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
        x = Dropout(0.1, name='dropout3')(x)

        if min_size // 24 >= 4:
            # Conv block 4
            x = Convolution2D(
                channel_size,
                3,
                1,
                padding='same',
                name='conv4')(x)
            x = BatchNormalization(axis=channel_axis, name='bn4')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
            x = Dropout(0.1, name='dropout4')(x)

        x = Reshape((-1, channel_size))(x)

        avg = GlobalAvgPool1D()(x)
        max = GlobalMaxPool1D()(x)
        x = concatenate([avg, max], axis=-1)
        # x = Dense(max(int(num_classes*1.5), 128), activation='relu', name='dense1')(x)
        x = Dropout(0.3)(x)
        outputs1 = Dense(num_classes, activation='softmax', name='output')(x)




        # bnorm_1 = BatchNormalization(axis=2)(inputs)
        lstm_1 = Bidirectional(CuDNNLSTM(64, name='blstm_1',
                                         return_sequences=True),
                               merge_mode='concat')(inputs)
        activation_1 = Activation('tanh')(lstm_1)
        dropout1 = SpatialDropout1D(0.5)(activation_1)
        attention_1 = Attention(8, 16)([dropout1, dropout1, dropout1])
        pool_1 = GlobalMaxPool1D()(attention_1)
        dropout2 = Dropout(rate=0.5)(pool_1)
        dense_1 = Dense(units=256, activation='relu')(dropout2)
        outputs2 = Dense(units=num_classes, activation='softmax')(dense_1)

        outputs = Average()([outputs1,outputs2])
        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0002,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, round_num, **kwargs):
        val_x, val_y = validation_data_fit
        if round_num >= 2:
            epochs = 10
        else:
            epochs = 5
        patience = 2

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)]

        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
