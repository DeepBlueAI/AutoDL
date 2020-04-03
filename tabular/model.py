import os
try:
    import skmultilearn
except:
    os.system('pip3 install scikit-multilearn')

import logging
# import sys
# sys.path.append("./AutoDL_sample_code_submission/tabular")
# print(os.getcwd())
from model_manger import LabelPowersetLGB, SklearnLGB, SVM, BinaryRelevances
from model_manger import ClassifierChains, BinaryRelevancesSimple, LabelPowersetXGB,LabelPowersetLGBN,BinaryRelevancesLGBN
import numpy as np
import pandas as pd

import sys
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import random

def auc_metric(solution, prediction, task='binary.classification'):
    '''roc_auc_score() in sklearn is fast than code provided by sponsor
    '''
    if solution.sum(axis=0).min() == 0 :
        return np.nan
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc*2-1)


class linear_stack(object):
    '''
    Linear stacking for multi classification task.
    '''
    def __init__(self, feature_size, iters = 100, rate = 0.02, verbose = 0, col_rate = 0.75):
        '''
            Parameters:
                feature_size: int
                    Number of weights.
                verbose: 0 or 1
                    If 1, the eval metric on the train set is printed at each iteration.
                    If 0, the eval metrci will not be printed.
        '''
        self.W = [1.0/feature_size]*feature_size # Weight init. self.W is a list.  
        self.iters = iters
        self.rate = rate
        self.verbose = verbose
        self.coef_ = None
        self.col_rate = col_rate
        self.metric = self.cal_auc
    
        self.pre_sco = -1
        
    
    def fit(self, X, Y):
        '''
            Parameters:
                X: Tensor (3-D Array). 
                    The first dim is feature size. The second dim is sample number. 
                    The third num is class number.
                Y: Array, on-hot encoded label
                    The first dim is sample number. The second dim is class number.
        '''

        for it in range(self.iters):
            arr = np.arange(len(self.W)) # Order of parameters optimization
            np.random.shuffle(arr) # Shuffle the order in each iteration
            for i in arr:
                score0 = self.metric(X, Y, self.W) # self defined metric

                w1 = self.W[:] # Deep copy of weights
                if w1[i] > self.rate:
                    w1 = self.sub_wei(w1, i, self.rate)
                    score1 = self.metric(X, Y, w1)
                    if score1 > score0: 
                        self.W = w1[:]
                        score0 = score1
                
                w2 = self.W[:] # Deep copy of weights
                w2 = self.add_wei(w2, i, self.rate)
                score2 = self.metric(X, Y, w2)
                if score2 > score0:
                    self.W = w2[:]
                    score0 = score2
            
            if self.verbose == 1:
                print ( '--- iter: ', it, '\tmetric:', score0, self.W )

            if abs(self.pre_sco - score0) < 0.0002:
                return
            self.pre_sco = score0
            
        self.coef_ = self.W

    def cal_auc(self, X, Y, coef):
        '''
            Parameters:
                X: Tensor (3-D Array). 
                    The first dim is feature size. The second dim is sample number. 
                    The third num is class number.
                Y: Array, on-hot encoded label
                    The first dim is sample number. The second dim is class number.
                coef: list
                    Weights.
        '''
        coef = np.array(coef)
        coef = coef/coef.sum()
        preds = X[0]*coef[0]
        for i in range(1, len(coef)):
            preds = preds + X[i]*coef[i]
        
        return auc_metric(Y, preds)
    
    def sub_wei(self, w, i, rate=0.003):
        '''# The i-th weight minus rate. And other weights add rate/number
        '''
        w[i] -= rate
        r = rate/(len(w)-1)
        for j in range(len(w)):
            if j != i:
                w[j] += r
        return w

    def add_wei(self, w, i, rate = 0.003):
        '''# The i-th weight add rate. And other weights minus rate/number
        '''
        w[i] += rate
        r = rate/(len(w) - 1)
        for j in range(len(w)):
            if j != i:
                w[j] -= r
        return w    

    
    def predict(self, X):
        '''
            Parameters:
                X: Tensor (3-D Array). 
                    The first dim is feature size. The second dim is sample number. 
                    The third num is class number.
        '''
        coef = np.array(self.W)
        coef = coef/coef.sum()
        preds = X[0]*coef[0]
        for i in range(1, len(coef)):
            preds = preds + X[i]*coef[i]
        return preds    
    

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
        self.model = None #BinaryRelevances() #BinaryRelevancesSimple()  #BinaryRelevances()
        self.step = 0
        # self.lgb_round = self.model_lgb.grow_boost_round
        self.lgb_round = 10
        
        self.X = []
        self.Y = []
        self.next_element = []
        self.data_step = 0
        
        self.cand_models = ['LGB'] + ['LGBN']*5   #['LGB','RF']
        self.best_val_res = [0]*30
        self.best_test_res = [0]*30

        self.model_num = -2
        
        self.n_lgb = 0
        
        self.X_train = None
        #self.X_val = None
        self.y_train = None
        #self.y_val = None

        self.new_start = True
        self.task_type = 'softmax'
        
        self.best_sco = -1
        self.best_res = []
        #self.val_res = []
        self.test_res = []
        #self.his_scos = []
    
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

        # print(f'dataset: {dataset}')
        t1 = time.time()

        if self.step == 0:
            #dataset = dataset.shuffle(buffer_size=10000000).batch(512)  #
            dataset = dataset.batch(512)  #
            iterator = dataset.make_one_shot_iterator()
            self.next_element = iterator.get_next()
                
        # Count examples on training set
        if not hasattr(self, 'num_examples_train'):
            logger.info("Counting number of examples on train set.")
            print ('dataset:',dataset)

#             X = []
#             Y = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        example, labels = sess.run(self.next_element)
                        example = np.squeeze(example)
                        self.X.extend(example)
                        self.Y.extend(labels)
                        self.data_step += 1
                        if self.data_step in [2, 6]:
                            break
                    except tf.errors.OutOfRangeError:
                        self.data_step = -1
                        break
                        
            self.X_train = np.array(self.X)
            self.y_train = np.array(self.Y)
            judge_y = np.sum(self.Y, axis=1)
            
            if self.step == 0:
                if (judge_y==1).all():
                    print('softmax')
                    self.task_type = 'softmax'
                    self.model = LabelPowersetLGB()
                else:
                    self.task_type = 'BinaryRelevances'
                    print(self.y_train[judge_y!=1])
                    self.model = BinaryRelevances(len(self.y_train[0])) #BinaryRelevancesSimple()  #BinaryRelevances()
                    print('BinaryRelevances')
                
            print('self.X_train.shape: {}'.format(self.X_train.shape))
            print('self.y_train.shape: {}.'.format(self.y_train.shape))
            
            if self.data_step == -1:
                self.num_examples_train = len(self.y_train)
                logger.info("Finished counting. There are {} examples for training set." \
                            .format(self.num_examples_train))
                
        print('train to numpy spand time: {}'.format(time.time()-t1))
        
        print ('###self.model_num:', self.model_num)
        print ('###step:', self.step)
            
        train_start = time.time()
        if self.model_num in [-2,-1]:
            self.new_start = False
            self.model_num += 1
            if hasattr(self, 'test_duration'):
                round = int(50*self.test_duration+5)
                self.model.set_grow_step(round)
                self.lgb_round += round

            self.model.fit(self.X_train, self.y_train)
            #self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)
        else:
            if self.cand_models[self.model_num] == 'LGB':
                if self.step == 2:
                    self.new_start = False
#                     self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.15, random_state=666)
                if hasattr(self, 'test_duration'):
                    round = int(50*self.test_duration+5)
                    self.model.set_grow_step(round)
                    self.lgb_round += round
                self.model.fit(self.X_train, self.y_train)
                
                if self.lgb_round >= 300 or self.step > 25:
                    self.new_start = True
                    #self.model_num += 1
                    #return
                
            elif self.cand_models[self.model_num] == 'LGBN':
                X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=random.randint(0,99999999))
                num_leaves = random.randint(26,35)
                bagging_fraction = random.uniform(0.8, 0.95)
                feature_fraction = random.uniform(0.85, 0.95)
                print ('#num_leaves, bagging_fraction, feature_fraction', num_leaves, bagging_fraction, feature_fraction)
                if self.task_type == 'softmax':
                    print ('LGBN model（softmax）...')
                    #num_leaves, bagging_fraction, feature_fraction
                    
                    self.model = LabelPowersetLGBN(num_leaves, bagging_fraction, feature_fraction)
                    self.model.fit(X_train, y_train) #y_train = np.argmax(y_train, axis=1)   self.y_train
                else:
                    print ('LGBN model（BinaryRelevances）...')
                    #self.model = BinaryRelevancesSimple('RF')
                    #self.model = RandomForestClassifier(n_estimators=200, max_depth=12)
                    self.model = BinaryRelevancesLGBN(len(y_train[0]), num_leaves, bagging_fraction, feature_fraction)
                    self.model.fit(X_train, y_train) #y_train = np.argmax(y_train, axis=1)   self.y_train
                    
                self.new_start = True
                
        train_end = time.time()
        # Update for time budget managing
        train_duration = train_end - train_start
        logger.info("{} step. {:.2f} sec used. ".format(self.step, train_duration))
        
        self.step += 1
        # self.done_training = True
    

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
        if self.done_training:
            return self.best_res
        
        start1 = time.time()
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
            # self.X_test = pd.DataFrame(self.X_test)
            # self.X_test.to_csv(r'D:\code\automl\neurips2019_autodl\AutoDL_public_data'+f'test{self.X_train.shape[0]}.csv')
            # print(1/0)
            # print(f'self.X_test.shape: {self.X_test.shape}')
            self.num_examples_test = self.X_test.shape[0]
            logger.info("Finished counting. There are {} examples for test set." \
                        .format(self.num_examples_test))
        print ('###test to_numpy time:', time.time() - start1)
        
        
        test_begin = time.time()
        logger.info("Begin testing...")
        
        if self.model_num < 1:
            self.best_res = self.model.predict(self.X_test)
            
            if self.new_start:
                self.model_num += 1
                self.test_res.append(self.best_res*2)
        else:
            self.model_num += 1
            pred = self.model.predict_proba(self.X_test)
            self.test_res.append(pred)
            
            print ('###update best result...')
            
            print ('###self.test_res', self.test_res)
            
            print ('begin ensemble...', self.best_res)
            #self.best_res = self.ensemble()
            self.best_res = np.mean(self.test_res, axis=0)
            print ('end ensemble...', self.best_res)
            
            #self.best_res = -1
        
        print('###self.best_res:', self.best_res)
        test_end = time.time()
        # Update some variables for time management
        self.test_duration = test_end - test_begin
        logger.info("[+] Successfully made one prediction. {:.2f} sec used. " \
                    .format(self.test_duration) + \
                    "Duration used for test: {:2f}".format(self.test_duration))
        
        
        if self.model_num == len(self.cand_models):
            self.done_training = True
            
        return self.best_res



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