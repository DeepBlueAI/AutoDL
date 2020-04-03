"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
from numba import jit

from multiprocessing import Pool

here = os.path.dirname(os.path.abspath(__file__))
# model_dirs = ['',                       # current directory
#               'AutoCV/kakaobrain',      # AutoCV/AutoCV2 winner model
#               #'AutoNLP/upwind_flys',    # AutoNLP 2nd place winner
#               'AutoNLP/DeepBlueAI',
#               'AutoSpeech/PASA_NJU',    # AutoSpeech winner
#               'tabular_Meysam']         # simple NN model
# for model_dir in model_dirs:
#   sys.path.append(os.path.join(here, model_dir))
#
#
# from AutoSpeech.PASA_NJU.model import Model as AutoSpeechModel
#
# #from AutoNLP.upwind_flys.model import Model as AutoNLPModel
# from AutoNLP.DeepBlueAI.model import Model as AutoNLPModel
#
# from tabular.model import Model as TabularModel
# from AutoCV.kakaobrain.model import Model as AutoCVModel
#
# DOMAIN_TO_MODEL = {'image': AutoCVModel,
#                    'video': AutoCVModel,
#                    'text': AutoNLPModel,
#                    'speech': AutoSpeechModel,
#                    'tabular': TabularModel}





model_dirs = ['',                       # current directory
              'AutoCV/kakaobrain',      # AutoCV winner model
              'AutoCV2/kakaobrain2',      # AutoCV2 winner model
              'AutoNLP/DeepBlueAI',    # AutoNLP 2nd place winner
              'AutoSpeech/PASA_NJU',    # AutoSpeech winner
              'tabular']         # simple NN model
for model_dir in model_dirs:
  sys.path.append(os.path.join(here, model_dir))
from AutoSpeech.PASA_NJU.model import Model as AutoSpeechModel
# from AutoNLP.upwind_flys.model import Model as AutoNLPModel

from AutoNLP.DeepBlueAI.model import Model as AutoNLPModel
# from tabular.model import Model as TabularModel
from tabular.model import Model as TabularModel

from AutoCV.kakaobrain.model import Model as AutoCVModel
from AutoCV2.kakaobrain2.model2 import Model2 as AutoCVModel2

DOMAIN_TO_MODEL = {'image': AutoCVModel,
                   'video': AutoCVModel2,
                   'text': AutoNLPModel,
                   'speech': AutoSpeechModel,
                   'tabular': TabularModel}



def my_token(x):
    x = x.reshape(-1)
    x = x[x > 0].astype(dtype = np.int)
    tokens = [index_to_token[j] for j in x]
    document = sep.join(tokens)
    return document

class Model():
  """A model that combine all winner solutions. Using domain inferring and
  apply winner solution in the corresponding domain."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata
    self.domain = infer_domain(metadata)
    logger.info("The inferred domain of current dataset is: {}."\
                .format(self.domain))
    self.domain_metadata = get_domain_metadata(metadata, self.domain)
    DomainModel = DOMAIN_TO_MODEL[self.domain]
    self.domain_model = DomainModel(self.domain_metadata)

    self.update_cnt = 0
    self.train_X = None
    self.test_X = None
    self.label = None
    self.sep = ''

    self.tra_X = []
    self.tra_Y = []
    self.max_dup = 2
    self.next_element = None

    self.to_word = False
    self.max_length = 1000
    self.mini = True

  def train(self, dataset, remaining_time_budget=None):
    """Train method of domain-specific model."""
    # Convert training dataset to necessary format and
    # store as self.domain_dataset_train



    if self.domain =='speech':
      self.domain_model.train(dataset,
                              remaining_time_budget=remaining_time_budget)
    else:
        start = time.time()
        self.set_domain_dataset(dataset, is_training=True)
        print ('###train set_domain_dataset(s):', time.time() - start)

        # Train the model
        self.domain_model.train(self.domain_dataset_train,
                                remaining_time_budget=remaining_time_budget)


    # Update self.done_training
    self.done_training = self.domain_model.done_training

  def test(self, dataset, remaining_time_budget=None):
    """Test method of domain-specific model."""
    # Convert test dataset to necessary format and
    # store as self.domain_dataset_test

    start = time.time()
    self.set_domain_dataset(dataset, is_training=False)
    print ('###test set_domain_dataset(s):', time.time() - start)


    # As the original metadata doesn't contain number of test examples, we
    # need to add this information
    if self.domain in ['text', 'speech'] and\
       (not self.domain_metadata['test_num'] >= 0):
      self.domain_metadata['test_num'] = len(self.X_test)

    # Make predictions
    if self.domain == 'text':
        Y_pred, self.to_word = self.domain_model.test(self.domain_dataset_test,
                                    remaining_time_budget=remaining_time_budget)
    else:
        # Make predictions
        Y_pred = self.domain_model.test(self.domain_dataset_test,
                                        remaining_time_budget=remaining_time_budget)

    self.update_cnt += 1
    # Update self.done_training
    self.done_training = self.domain_model.done_training

    return Y_pred


  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  def to_numpy(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    if is_training:
      subset = 'train'
    else:
      subset = 'test'

    attr_X = 'X_{}'.format(subset)
    attr_Y = 'Y_{}'.format(subset)

    # Only iterate the TF dataset when it's not done yet
    if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
# #       max_elems = np.iinfo(np.int64).max
      dataset = dataset.padded_batch(128, padded_shapes=([None,1,1,1], [None]), padding_values=(tf.constant(-1, dtype=tf.float32)
                                                 ,tf.constant(-1, dtype=tf.float32)))

      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      X = []
      Y = []

      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # 开启一个协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            while not coord.should_stop():
                example, labels = sess.run(next_element)
                X.extend(example)
                Y.extend(labels)
        except tf.errors.OutOfRangeError:  #如果读取到文件队列末尾会抛出此异常
            print("done! now lets kill all the threads……")
        finally:
            # 协调器coord发出所有线程终止信号
            coord.request_stop()
            print('all threads are asked to stop!')
        coord.join(threads) #把开启的线程加入主线程，等待threads结束
        print('all threads are stopped!')

      setattr(self, attr_X, X)
      setattr(self, attr_Y, Y)

#     X = getattr(self, attr_X)
#     Y = getattr(self, attr_Y)
    return np.array(X), Y


  def train_to_numpy(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    if is_training:
      subset = 'train'
    else:
      subset = 'test'

    attr_X = 'X_{}'.format(subset)
    attr_Y = 'Y_{}'.format(subset)

    X = []
    Y = []


    batch_size = 128
    N = batch_size

    if self.update_cnt == 0:
        #-----------
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None,1,1,1], [None]), padding_values=(tf.constant(-1, dtype=tf.float32)
                                                 ,tf.constant(-1, dtype=tf.float32)))
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            example, labels = sess.run(self.next_element)
#             X.extend(example)
#             Y.extend(labels)
            N = np.sum(labels[:,0])
        print ('###labels:', labels[:5])
        #-----------

        if N == batch_size:
            print ('shuffle...')
            iterator = dataset.shuffle(1024).shuffle(10000000).make_one_shot_iterator()
            self.next_element = iterator.get_next()
            N = -1

    print ('###N:batch_size', N, batch_size)
    # Only iterate the TF dataset when it's not done yet
#     if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
#         with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
#             while True:
#                 try:
#                     example, labels = sess.run(self.next_element)
#                     X.extend(example)
#                     Y.extend(labels)
#                     if N != batch_size and len(Y) > 1000:
#                         print ('###small data...')
#                         self.max_dup = 3
#                         break
#                 except tf.errors.OutOfRangeError:
#                     setattr(self, attr_X, X)
#                     setattr(self, attr_Y, Y)
#                     break


    if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # 开启一个协调器
            coord = tf.train.Coordinator()
            # 使用start_queue_runners 启动队列填充
            threads = tf.train.start_queue_runners(sess, coord)

            try:
                while not coord.should_stop():
                    example, labels = sess.run(self.next_element)
                    X.extend(example)
                    Y.extend(labels)
                    if self.mini and N != batch_size and len(Y) > 2000:
                        self.mini = False
                        t = np.sum(Y, axis=0)
                        print ('###t0:', t)

                        t = len(t)*t/np.sum(t)

                        nan_rate = np.sum(t<0.032)/len(t)
                        print ('###nan_rate:', nan_rate)
                        if nan_rate < 0.16:
                            print ('###small data...')
                            self.max_dup = 3

                            for i in range(len(t)):
                                if t[i] == 0:
                                    Y[i] = np.zeros(len(Y[0]))
                                    Y[i][i] = 1

                                    Y[i+150] = np.zeros(len(Y[0]))
                                    Y[i+150][i] = 1

                                    Y[i+500] = np.zeros(len(Y[0]))
                                    Y[i+500][i] = 1

                                    Y[i+900] = np.zeros(len(Y[0]))
                                    Y[i+900][i] = 1
                            t = np.sum(Y, axis=0)
                            print ('###t1:', t)
                            break
            except tf.errors.OutOfRangeError:  #如果读取到文件队列末尾会抛出此异常
                print("done! now lets kill all the threads……")
                setattr(self, attr_X, X)
                setattr(self, attr_Y, Y)
            finally:
                # 协调器coord发出所有线程终止信号
                coord.request_stop()
                print('all threads are asked to stop!')

            coord.join(threads) #把开启的线程加入主线程，等待threads结束
            print('all threads are stopped!')


#     X = getattr(self, attr_X)
#     Y = getattr(self, attr_Y)
    return np.array(X), Y



  def set_domain_dataset(self, dataset, is_training=True):
    """Recover the dataset in corresponding competition format (esp. AutoNLP
    and AutoSpeech) and set corresponding attributes:
      self.domain_dataset_train
      self.domain_dataset_test
    according to `is_training`.
    """
    if is_training:
      subset = 'train'
    else:
      subset = 'test'
    attr_dataset = 'domain_dataset_{}'.format(subset)

    if (not hasattr(self, attr_dataset)) or (self.domain == 'text' and self.update_cnt < self.max_dup) or (self.domain == 'text' and self.to_word): #2
      logger.info("Begin recovering dataset format in the original " +
                  "competition for the subset: {}...".format(subset))


      if self.domain == 'text':
        # Get X, Y as lists of NumPy array

        if self.max_dup == 3 and subset == 'test' and self.update_cnt == 1:
            return

        if self.update_cnt == 0 or (self.max_dup == 3 and subset == 'train' and self.update_cnt == 1):
            start = time.time()

            if subset == 'train':
                X, Y = self.train_to_numpy(dataset, is_training=is_training)
            else:
                X, Y = self.to_numpy(dataset, is_training=is_training)

            print (subset,'###to_numpy set_domain_dataset(s):', time.time() - start)
            print ('###data size:', len(Y))

            # Get separator depending on whether the dataset is in Chinese
            if is_chinese(self.metadata):
              self.sep = ''
            else:
              self.sep = ' '

            start = time.time()
            for i in range(len(X)):
               X[i] = X[i].reshape(-1)
               X[i] = X[i][X[i] > 0].astype(dtype = np.int)
               #X[i] = X[i][X[i] > 0].astype(dtype = str)
            print ('###reshape(s):', time.time() - start)


            if subset == 'train':
                if self.update_cnt == 0:
                    self.tra_Y = Y
                    self.tra_X = X
                else:
                    X = np.concatenate((X, self.tra_X), axis=0)
                    Y = Y + self.tra_Y

            if is_training:
                self.label = Y
                self.train_X = X
            else:
                self.test_X = X
            corpus = []
            corpus = X
            print ('###index_to_token(s) for svm:', time.time() - start)
        else: #for DNN
            # Retrieve vocabulary (token to index map) from metadata and construct
            # the inverse map

            start = time.time()

            if self.to_word == False:
                #deal 1
                print ('deal 1')
                num_sentence = 0
                if is_training:
                    X = self.train_X
                    num_sentence = len(X)
                    text_lens = np.zeros( len(X), dtype=np.int32 )
                    for i in range(len(X)):
                        text_lens[i] = len(X[i])
                    self.max_length = np.sort(text_lens)[int(num_sentence*0.92)] #0.95
                else:
                    X = self.test_X
                    num_sentence = len(X)

                corpus = np.zeros((num_sentence, self.max_length), dtype=np.int32)
                for i in range(len(X)):
                    n = min(len(X[i]), self.max_length)
                    corpus[i][:n] = X[i][:n]

                if is_training:
                    self.train_X = corpus
                else:
                    self.test_X = corpus
                print ('MAX_SEQ_LENGTH1:', self.max_length)
            else:
                #deal 2
                print ('deal 2')
                vocabulary = self.metadata.get_channel_to_index_map()
                index_to_token = [None] * len(vocabulary)
                for token in vocabulary:
                    index = vocabulary[token]
                    index_to_token[index] = token

                corpus = []
                if is_training:
                    X = self.train_X
                else:
                    X = self.test_X

                for x in X: # each x in X is a list of indices (but as float)
                    tokens = [index_to_token[int(i)] for i in x.astype(dtype = np.int) ]
                    document = self.sep.join(tokens)
                    corpus.append(document)
            #------------------------------------
            print ('###index_to_token(s) for dnn:', time.time() - start)


        # Construct the dataset for training or test
        if is_training:
          labels = np.array(self.label)
          domain_dataset = corpus, labels
        else:
          domain_dataset = corpus

        # Set the attribute
        setattr(self, attr_dataset, domain_dataset)

      elif self.domain == 'speech':
        # Get X, Y as lists of NumPy array
        X, Y = self.to_numpy(dataset, is_training=is_training)

        # Convert each array to 1-D array
        X = [np.squeeze(x) for x in X]

        # Construct the dataset for training or test
        if is_training:
          labels = np.array(Y)
          domain_dataset = X, labels
        else:
          domain_dataset = X

        # Set the attribute
        setattr(self, attr_dataset, domain_dataset)

      elif self.domain in ['image', 'video', 'tabular']:
        setattr(self, attr_dataset, dataset)
      else:
        raise ValueError("The domain {} doesn't exist.".format(self.domain))


def infer_domain(metadata):
  """Infer the domain from the shape of the 4-D tensor.

  Args:
    metadata: an AutoDLMetadata object.
  """
  row_count, col_count = metadata.get_matrix_size(0)
  sequence_size = metadata.get_sequence_size()
  channel_to_index_map = metadata.get_channel_to_index_map()
  domain = None
  if sequence_size == 1:
    if row_count == 1 or col_count == 1:
      domain = "tabular"
    else:
      domain = "image"
  else:
    if row_count == 1 and col_count == 1:
      if len(channel_to_index_map) > 0:
        domain = "text"
      else:
        domain = "speech"
    else:
      domain = "video"
  return domain


def is_chinese(metadata):
  """Judge if the dataset is a Chinese NLP dataset. The current criterion is if
  each word in the vocabulary contains one single character, because when the
  documents are in Chinese, we tokenize each character when formatting the
  dataset.

  Args:
    metadata: an AutoDLMetadata object.
  """
  domain = infer_domain(metadata)
  if domain != 'text':
    return False

  cnt = 0
  for i, token in enumerate(metadata.get_channel_to_index_map()):
    if len(token) == 1:
      cnt += 1
    if i >= 300:
      break

  if cnt > 150:
      print ('ZH!!!')
      return True
  else:
      print ('EN!!!')
      return False

#   for i, token in enumerate(metadata.get_channel_to_index_map()):
#     if len(token) != 1:
#       return False
#     if i >= 300:
#       break
#  return True


def get_domain_metadata(metadata, domain, is_training=True):
  """Recover the metadata in corresponding competitions, esp. AutoNLP
  and AutoSpeech.

  Args:
    metadata: an AutoDLMetadata object.
    domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
  """
  if domain == 'text':
    # Fetch metadata info from `metadata`
    class_num = metadata.get_output_size()
    num_examples = metadata.size()
    language = 'ZH' if is_chinese(metadata) else 'EN'
    time_budget = 1200 # WARNING: Hard-coded

    # Create domain metadata
    domain_metadata = {}
    domain_metadata['class_num'] = class_num
    if is_training:
      domain_metadata['train_num'] = num_examples
      domain_metadata['test_num'] = -1
    else:
      domain_metadata['train_num'] = -1
      domain_metadata['test_num'] = num_examples
    domain_metadata['language'] = language
    domain_metadata['time_budget'] = time_budget

    return domain_metadata
  elif domain == 'speech':
    # Fetch metadata info from `metadata`
    class_num = metadata.get_output_size()
    num_examples = metadata.size()

    # WARNING: hard-coded properties
    file_format = 'wav'
    sample_rate = 16000

    # Create domain metadata
    domain_metadata = {}
    domain_metadata['class_num'] = class_num
    if is_training:
      domain_metadata['train_num'] = num_examples
      domain_metadata['test_num'] = -1
    else:
      domain_metadata['train_num'] = -1
      domain_metadata['test_num'] = num_examples
    domain_metadata['file_format'] = file_format
    domain_metadata['sample_rate'] = sample_rate

    return domain_metadata
  else:
    return metadata


def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
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
