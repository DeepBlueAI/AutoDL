import logging
from model_manger import LGBMultiClass
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import time


class Model(object):
    """Fully connected neural network with no hidden layer."""

    def __init__(self, metadata):
        """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
        self.done_training = False
        self.metadata = metadata
        self.output_dim = self.metadata.get_output_size()
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
        self.model = MLkNN(k=20)
        self.step = 0
        self.lgb_round = 80

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.
    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
        if self.done_training:
            return
        self.step += 1
        # print(f'dataset: {dataset}')
        t1 = time.time()
        # Count examples on training set
        if not hasattr(self, 'num_examples_train'):
            logger.info("Counting number of examples on train set.")
            dataset = dataset.batch(128)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            X = []
            Y = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        example, labels = sess.run(next_element)
                        example = np.squeeze(example)
                        X.extend(example)
                        Y.extend(labels)
                    except tf.errors.OutOfRangeError:
                        break
            self.X_train = np.array(X)
            self.y_train = np.array(Y)
            print('self.X_train.shape: {}'.format(self.X_train.shape))
            print('self.y_train.shape: {}.'.format(self.y_train.shape))
            self.num_examples_train = len(self.y_train)
            logger.info("Finished counting. There are {} examples for training set." \
                        .format(self.num_examples_train))
        print('spand time: {}'.format(time.time()-t1))
        if self.lgb_round >= 300 or self.step > 10:
            self.done_training = True
            return
        if hasattr(self, 'test_duration'):
            round = int(50*self.test_duration+5)
            self.lgb_round += round
        train_start = time.time()
        self.X_train = self.imputer.fit_transform(self.X_train)
        self.model.fit(self.X_train, self.y_train)
        train_end = time.time()

        # Update for time budget managing
        train_duration = train_end - train_start
        logger.info("{} step. {:.2f} sec used. ".format(self.step, train_duration))

        self.done_training = True

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
        # Count examples on test set
        if not hasattr(self, 'num_examples_test'):
            logger.info("Counting number of examples on test set.")
            dataset = dataset.batch(128)
            iterator = dataset.make_one_shot_iterator()
            example, labels = iterator.get_next()
            X = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        ex = sess.run(example)
                        ex = np.squeeze(ex)
                        X.extend(ex)
                    except tf.errors.OutOfRangeError:
                        break
            self.X_test = np.array(X)
            self.num_examples_test = self.X_test.shape[0]
            logger.info("Finished counting. There are {} examples for test set." \
                        .format(self.num_examples_test))

        test_begin = time.time()
        logger.info("Begin testing...")
        self.X_test = self.imputer.fit_transform(self.X_test)
        predictions = self.model.predict(self.X_test).A
        # print(type(predictions))
        # print(predictions.A)
        # preds = self.model.predict_proba(self.X_test)
        # print(preds)
        # test_results = pd.Series(test_results).map(self.remps).values
        # predictions = self.bin2y(test_results)
        # print(predictions)
        test_end = time.time()
        # Update some variables for time management
        self.test_duration = test_end - test_begin
        logger.info("[+] Successfully made one prediction. {:.2f} sec used. " \
                    .format(self.test_duration) + \
                    "Duration used for test: {:2f}".format(self.test_duration))
        return predictions

    def y2bin(self, y):
        res = y[:, 0]
        for i in range(1, y.shape[1]):
            res *= 2
            res += y[:, i]
        return res

    def bin2y(self, bin):
        y = np.array([bin%2]).T
        i = 1
        while i < self.output_dim:
            i += 1
            bin = bin//2
            y = np.c_[np.array([bin%2]).T, y]
            # y = np.insert(y, 0, values=bin%2, axis=1)
        return y



def get_logger(verbosity_level):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model_dnn.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger('INFO')