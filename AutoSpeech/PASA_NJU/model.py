#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from .data_manager import DataManager
from .model_manager import ModelManager
from tools import log, timeit
import numpy as np
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)



class DataIter(object):
        def __init__(self,dataset):
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

            self.iterator = dataset.make_one_shot_iterator()
            self.next_element = self.iterator.get_next()
        @timeit
        def get_data(self,nums):
            X = []
            Y = []
            for i in range(nums):
                try:
                    example, labels = self.sess.run(self.next_element)
                    X.append(example)
                    Y.append(labels)
                    # i+=1
                except tf.errors.OutOfRangeError:
                    break

            X = [np.squeeze(x) for x in X]
            labels = np.array(Y)
            return X, labels


class Model(object):
    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_loop_num = 0
        log('Metadata: {}'.format(self.metadata))
        self.dataiter = None
        self.data_manager = None
        self.model_manager = None

        self.train_output_path = train_output_path
        self.test_input_path = test_input_path
        self.nums = [0]*4
        self.train_x = []
    @timeit
    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.

        :param train_dataset: tuple, (train_x, train_y)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        self.train_loop_num += 1
        class_num = self.metadata['class_num']
        train_num = self.metadata['train_num']
        self.nums[0] = int(train_num*0.07)
        self.nums[1] = int(train_num*0.28 - self.nums[0])
        self.nums[2] = int(train_num*0.77 - self.nums[1]-self.nums[0])
        self.nums[3] = int(train_num - self.nums[2]- self.nums[1]-self.nums[0])


        if self.train_loop_num == 1:
            if self.dataiter is None:
                self.dataiter = DataIter(train_dataset)
            tem_x,tem_y = self.dataiter.get_data(self.nums[self.train_loop_num-1])
            print('load data num',len(tem_x))
            self.data_manager = DataManager(self.metadata, tem_x,tem_y)
            self.model_manager = ModelManager(self.metadata, self.data_manager)

        if self.train_loop_num == 2:
            tem_x, tem_y = self.dataiter.get_data(self.nums[self.train_loop_num-1])
            print('load data num',len(tem_x))
            self.data_manager._all_x = list(self.data_manager._all_x)
            self.data_manager._all_y = list(self.data_manager._all_y)
            self.data_manager._all_x.extend(tem_x)
            self.data_manager._all_y.extend(tem_y)

            self.data_manager._all_x = np.array(self.data_manager._all_x)
            self.data_manager._all_y = np.array(self.data_manager._all_y)
        if self.train_loop_num == 3:
            tem_x, tem_y = self.dataiter.get_data(self.nums[self.train_loop_num-1])
            print('load data num',len(tem_x))
            self.data_manager._all_x = list(self.data_manager._all_x)
            self.data_manager._all_y = list(self.data_manager._all_y)
            self.data_manager._all_x.extend(tem_x)
            self.data_manager._all_y.extend(tem_y)

            self.data_manager._all_x = np.array(self.data_manager._all_x)
            self.data_manager._all_y = np.array(self.data_manager._all_y)
        if self.train_loop_num == 4:
            tem_x, tem_y = self.dataiter.get_data(self.nums[self.train_loop_num-1])
            print('load data num',len(tem_x))
            self.dataiter.sess.close()
            self.data_manager._all_x = list(self.data_manager._all_x)
            self.data_manager._all_y = list(self.data_manager._all_y)
            self.data_manager._all_x.extend(tem_x)
            self.data_manager._all_y.extend(tem_y)

            self.data_manager._all_x = np.array(self.data_manager._all_x)
            self.data_manager._all_y = np.array(self.data_manager._all_y)
        self.model_manager.fit(train_loop_num=self.train_loop_num)

        if self.train_loop_num > 500:
            self.done_training = True

    @timeit
    def test(self, test_x, remaining_time_budget=None):
        """
        :param test_x: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        # extract test feature
        pred_y = self.model_manager.predict(test_x, is_final_test_x=True, train_loop_num=self.train_loop_num)

        result = pred_y
        return result


@timeit
def set_domain_dataset(metadata,dataset,train_loop_num):
    """Recover the dataset in corresponding competition format (esp. AutoNLP
    and AutoSpeech) and set corresponding attributes:
      self.domain_dataset_train
      self.domain_dataset_test
    according to `is_training`.
    """
    print(metadata)



    # input()
    if train_loop_num==0:
        class_num = metadata['class_num']
        train_num = metadata['train_num']
        class_num_dict = {}

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X = []
        Y = []
        i=0
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            while True:
                try:
                    example, labels = sess.run(next_element)
                    X.append(example)
                    Y.append(labels)

                    l = np.argmax(labels)
                    if l in class_num_dict:
                        class_num_dict[l]+=1
                    else:
                        class_num_dict[l] = 1
                    if i>=(train_num*0.2):
                        break

                    if i>=(train_num*0.16
                    ):
                        stop = True
                        for k in class_num_dict:
                            if class_num_dict[k]<=1:
                                stop = False
                        if stop == True:
                            break
                    i+=1
                except tf.errors.OutOfRangeError:
                    break
        print('load data index ',i)
        X = [np.squeeze(x) for x in X]

        labels = np.array(Y)
        domain_dataset = X, labels
        # Set the attribute
        return domain_dataset
        pass

    else:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X = []
        Y = []
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            while True:
                try:
                    example, labels = sess.run(next_element)
                    X.append(example)
                    Y.append(labels)
                except tf.errors.OutOfRangeError:
                    break
        X = [np.squeeze(x) for x in X]
        labels = np.array(Y)
        domain_dataset = X, labels
        # Set the attribute
        return domain_dataset



if __name__ == '__main__':
    from AutoDL_ingestion_program.dataset import AutoDLDataset

    D_train = AutoDLDataset('/home/user_bj2/p/dataset/speech/data02/data02.data/train')
    # print(D_train.get_metadata())
    m = Model(1)

    m.train(D_train.get_dataset())
    m.train(D_train.get_dataset())
    m.train(D_train.get_dataset())
    m.train(D_train.get_dataset())
