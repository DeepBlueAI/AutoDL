#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:12
# @Author:  Mecthew

from sklearn.linear_model import logistic
import numpy as np
import abc
class ModelBase(object):
    @abc.abstractmethod
    def init_model(self, **kwargs):
        pass
    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, test_x):
        pass
def ohe2cat(label):
    return np.argmax(label, axis=1)

class LogisticRegression(ModelBase):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        # clear_session()
        self.max_length = None
        self._model = None
        self.is_init = False

    def init_model(self,
                   config,
                   **kwargs):

        num_classes = config['num_classes']
        sample_num = config['sample_num']

        max_iter = 200
        C = 1.0
        self._model = logistic.LogisticRegression(
            C=C, max_iter=max_iter, solver='liblinear', multi_class='auto')
        self.is_init = True

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train, ohe2cat(y_train))

    def predict(self, x_test, batch_size=32):
        return self._model.predict_proba(x_test)




