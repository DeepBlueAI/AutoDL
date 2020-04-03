import multiprocessing
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from skmultilearn.adapt import MLkNN
from sklearn.calibration import CalibratedClassifierCV
import json
import time
from sklearn.ensemble import RandomForestClassifier

np.set_printoptions(suppress=True)
NUM_THREAD = 7 #multiprocessing.cpu_count() - 1


def get_log_lr(num_boost_round,max_lr,min_lr):
    learning_rates = [max_lr+(min_lr-max_lr)/np.log(num_boost_round)*np.log(i) for i in range(1,num_boost_round+1)]
    return learning_rates


class LabelPowersetLGB:
    def __init__(self):
        self.model = None
        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "None",
            "verbosity": 1,
            # "seed": 888,
            "num_threads": NUM_THREAD
        }

        self.hyperparams = {
            # 'num_class': num_class,
            # 'two_round': False,
            # 'num_leaves': 20, 'bagging_fraction': 0.9, 'bagging_freq': 3,
            # 'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
            # 'lambda_l2': 0.5, 'min_data_in_leaf': 50
        }

#         self.learning_rates = 0.05
#         self.max_boost_round = 500
        self.grow_boost_round = 10
        self.train_times = 1
        # self.train_size = 0
        
        self.lr = get_log_lr(160, 0.25, 0.015) + [0.015]*3000
        #self.lr = [0.035] * 3000 #0.035
        self.step = 0

        
    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

    def fit(self, X_train, y_train, categorical_feature=None):

        start = time.time()
        self.params['num_class'] = y_train.shape[1]
        y_train = np.argmax(y_train, axis=1)
        if not categorical_feature:
            categorical_feature = []

        if self.step < 2:
            self.model = None
        
        lr = self.lr[self.step*self.grow_boost_round:(self.step+1)*self.grow_boost_round]
        print ('###lr:', self.step, lr[:5])
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        params = self.params
        hyperparams = self.hyperparams
        self.model = lgb.train(params={**params, **hyperparams},
                               train_set=lgb_train, categorical_feature=categorical_feature,
                               init_model=self.model,
                               #valid_sets=lgb_train, valid_names='train',
                               num_boost_round=self.grow_boost_round,
                               learning_rates = lr,
                               # early_stopping_rounds=10,
                               verbose_eval=1,
                               )
        print ('training time...', time.time() - start)
        
        self.step += 1
        return self
    

    def predict(self, X_test):
        return self.model.predict(X_test)

    def y2bin(self, y):
        res = y[:, 0]
        self.output_dim = y.shape[1]
        for i in range(1, y.shape[1]):
            res *= 2
            res += y[:, i]
        return res


class LabelPowersetLGBN:
    def __init__(self, num_leaves, bagging_fraction, feature_fraction):
        self.model = None
        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "None",
            "verbosity": 1,
            # "seed": 888,
            "num_threads": NUM_THREAD
        }

        self.hyperparams = {
            'num_leaves': num_leaves,
            'bagging_fraction': bagging_fraction,
            'featur,e_fraction': feature_fraction,
            # 'num_class': num_class,
            # 'two_round': False,
            # 'num_leaves': 20, 'bagging_fraction': 0.9, 'bagging_freq': 3,
            # 'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
            # 'lambda_l2': 0.5, 'min_data_in_leaf': 50
        }

        self.grow_boost_round = 10
        self.train_times = 1
        # self.train_size = 0
        
        self.lr = get_log_lr(160, 0.25, 0.015) + [0.015]*3000
        #self.lr = [0.035] * 3000 #0.035
        self.step = 0

    def fit(self, X_train, y_train, categorical_feature=None):

        start = time.time()
        self.params['num_class'] = y_train.shape[1]
        y_train = np.argmax(y_train, axis=1)
        if not categorical_feature:
            categorical_feature = []

        if self.step < 2:
            self.model = None
        
        num_boost_round = 230
        lr = self.lr[0:num_boost_round]
        print ('###lr:', self.step, lr[:5])
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        params = self.params
        hyperparams = self.hyperparams
        self.model = lgb.train(params={**params, **hyperparams},
                               train_set=lgb_train, categorical_feature=categorical_feature,
                               #init_model=self.model,
                               #valid_sets=lgb_train, valid_names='train',
                               num_boost_round=num_boost_round,
                               learning_rates = lr,
                               # early_stopping_rounds=10,
                               verbose_eval=1,
                               )
        print ('training time...', time.time() - start)
        
        self.step += 1
        return self
    

    def predict_proba(self, X_test):
        return self.model.predict(X_test)

    def y2bin(self, y):
        res = y[:, 0]
        self.output_dim = y.shape[1]
        for i in range(1, y.shape[1]):
            res *= 2
            res += y[:, i]
        return res
    
import xgboost as xgb    
class LabelPowersetXGB:
    def __init__(self):
        self.model = None
        self.params = {
            "booster": "gbtree",
            "objective": "multi:softprob",
            # "eval_metric": "None",
            "verbosity": 1,
            # "seed": 888,
            #'tree_method': 'gpu_hist',
            #'gpu_id': 0,
#             "nthread": NUM_THREAD
        }

        self.hyperparams = {
            'eta': 0.08,
        }
        self.learning_rates = 0.05
        self.max_boost_round = 500
        self.grow_boost_round = 5
        self.train_times = 1
        # self.lr = get_log_lr(160, 0.25, 0.015) + [0.015] * 3000
        self.step = 0
        # self.train_size = 0


    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

    def fit(self, X_train, y_train, categorical_feature=None):
        self.params['num_class'] = y_train.shape[1]
        y_train = np.argmax(y_train, axis=1)
        if not categorical_feature:
            categorical_feature = []
        # feats = list(range(X_train.shape[1]))
        lgb_train = xgb.DMatrix(X_train, label=y_train)
        params = self.params
        hyperparams = self.hyperparams
        self.model = xgb.train({**params, **hyperparams},
                               lgb_train,
                               xgb_model=self.model,
                               # evals = (lgb_train, 'train'), verbose_eval=20,
                               num_boost_round=self.grow_boost_round,
                               # early_stopping_rounds=10,
                               )
        return self

    def predict_proba(self, X_test):
        X_test = xgb.DMatrix(X_test)
        return self.model.predict(X_test)
        # return self.bin2pred(self.model.predict(X_test))
    


class SklearnLGB:
    def __init__(self):
        self.params = {
            # "boosting_type": "gbdt",
            "objective": "softmax",
            # "metric": 'None',
            "learning_rate": 0.05,
            "verbosity": 1,
            "seed": 888,
            "num_threads": NUM_THREAD
        }

    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

    def valid_fit(self, X_train, y_train):
        uni = int(np.max(y_train))+1
        self.params['num_class'] = uni
        # for i in range(10):
        #     print(','.join(X_train[i].astype(str)))
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=8)
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_test = lgb.Dataset(X_test, label=y_test)
        self.model = lgb.train(params=self.params,
                               train_set=lgb_train,
                               valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                               # feval=self.score,
                               num_boost_round=100,
                               early_stopping_rounds=10,
                               verbose_eval=10,
                               )
        # print(self.model.feature_name())
        # print(self.model.feature_importance())
        # print(self.model.predict(X_test))
        df_imp = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                               'importances': self.model.feature_importance()})

        print('importants: ', df_imp.sort_values('importances', ascending=False))

    def fit(self, X_train, y_train):
        y_train = self.y2bin(y_train)
        uni = list(np.unique(y_train))
        self.remps = {i: uni[i] for i in range(len(uni))}
        mps = {uni[i]: i for i in range(len(uni))}
        y_train = pd.Series(y_train).map(mps).values

        uni = int(np.max(y_train)) + 1
        self.params['num_class'] = uni

        uni = int(np.max(y_train)) + 1
        self.params['num_class'] = uni

        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params=self.params,
                               train_set=lgb_train,
                               valid_sets=lgb_train, valid_names='train',
                               # feval=self.score,
                               num_boost_round=100,
                               early_stopping_rounds=10,
                               verbose_eval=10,
                               )
        df_imp = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                               'importances': self.model.feature_importance()})

        print('importants: ', df_imp.sort_values('importances', ascending=False))

        
    def predict(self, X_test):
        test_results = np.argmax(self.model.predict(X_test), axis=1)
        return self.bin2y(pd.Series(test_results).map(self.remps).values)

    def y2bin(self, y):
        res = y[:, 0]
        self.output_dim = y.shape[1]
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


class BinaryRelevancesSimple2:
    def __init__(self):
        # self.params = {
        #     # 'num_class': num_class,
        #     # "boosting_type": "gbdt",
        #     "objective": "binary",
        #     "metric": 'None',
        #     "learning_rate": 0.05,
        #     "verbosity": 1,
        #     "seed": 888,
        #     "num_threads": NUM_THREAD
        # }
        self.model = BinaryRelevance(LGBMClassifier())

    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

    def fit(self, X_train, y_train):
        print ('###start trainging...')
        start = time.time()
        self.model.fit(X_train, y_train)
        print ('####training time:', time.time() - start)

    def predict(self, X_test):
        return self.model.predict_proba(X_test).A


class BinaryRelevancesSimple:
    def __init__(self, model):
        # self.params = {
        #     # 'num_class': num_class,
        #     # "boosting_type": "gbdt",
        #     "objective": "binary",
        #     "metric": 'None',
        #     "learning_rate": 0.05,
        #     "verbosity": 1,
        #     "seed": 888,
        #     "num_threads": NUM_THREAD
        # }

        self.model = BinaryRelevance(LGBMClassifier())
        if model == 'RF':
            self.model = BinaryRelevance(RandomForestClassifier(n_estimators=200, max_depth=12))

#     def set_grow_step(self, new_step):
#         self.grow_boost_round = new_step
     
    def fit(self, X_train, y_train):
        print ('###start trainging...')
        start = time.time()
        self.model.fit(X_train, y_train)
        print ('####training time:', time.time() - start)
 
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test).A
    
    
class BinaryRelevances:
    def __init__(self, class_num):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "None",
            "verbosity": 1,
            # "seed": 888,
            "num_threads": NUM_THREAD
        }
        self.hyperparams = {
            # 'two_round': False,
            'num_leaves': 28, 
            'max_depth':-1,
            'subsample_for_bin':200000,
#            'min_data_in_leaf': 50,
#             'lambda_l1': 0.5,
#             'lambda_l2': 0.5,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 3,
#             'bagging_fraction': 0.9,
            #'min_data_in_leaf': 50
            #'min_split_gain':0.0,
            #'bagging_fraction': 0.9, 'bagging_freq': 3,
            # 'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
            # 'lambda_l2': 0.5, 'min_data_in_leaf': 50
        }
        self.grow_boost_round = 10
        
        self.lr = get_log_lr(160, 0.23, 0.025) + [0.025]*3000
        self.step = 0
        self.models = [None]*class_num

#     def get_log_lr(self,num_boost_round,max_lr,min_lr):
#         learning_rates = [max_lr+(min_lr-max_lr)/np.log(num_boost_round)*np.log(i) for i in range(1,num_boost_round+1)]
#         return learning_rates


    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

        
    def fit(self, X_train, y_train, categorical_feature=None):
        
        start = time.time()
        
        class_num = y_train.shape[1]
        
        #self.models = [LGBMClassifier()]*class_num
        if not categorical_feature:
            categorical_feature = []
        
#         if self.step == 0:
#             self.grow_boost_round = 2
#         else:
#             self.grow_boost_round = 10
            
        print ('###categorical_feature:', categorical_feature)
        lr = self.lr[self.step*self.grow_boost_round:(self.step+1)*self.grow_boost_round]
        print ('###lr:', self.step, lr)
        
        
#         if self.step == 1:
#             self.models = [None]*class_num
        
        if self.step < 2:
            self.models = [None]*class_num
            
        for i in range(class_num):
            y_tmp = y_train[:, i]
            
            #self.models[i] = LGBMClassifier()
            #self.models[i].fit(X_train, y_tmp)
            
            lgb_train = lgb.Dataset(X_train, label=y_tmp)
            print ('###model I:', i)
            self.models[i] = lgb.train(
                               params={**self.params, **self.hyperparams},
                               train_set=lgb_train, categorical_feature=categorical_feature,
                               init_model=self.models[i],
                               #valid_sets=lgb_train, valid_names='train',
                               num_boost_round=self.grow_boost_round,
                               learning_rates = lr,
                               # early_stopping_rounds=10,
                               verbose_eval=2,
                               )
        self.step += 1
        print ('###trainging time:', time.time() - start)

        
    def predict(self, X_test):
        res = self.models[0].predict(X_test)
        #res = self.models[0].predict_proba(X_test)[:, 1]
        #print ('###res1:', res)
        for model in self.models[1:]:
            tmp = model.predict(X_test)
            #tmp = model.predict_proba(X_test)[:, 1]
            res = np.c_[res, tmp]
        #print ('###res2:', res)
        return res


    
class BinaryRelevancesLGBN:
    def __init__(self, class_num,num_leaves,bagging_fraction,feature_fraction):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "None",
            "verbosity": 1,
            # "seed": 888,
            "num_threads": NUM_THREAD
        }
        self.hyperparams = {
            # 'two_round': False,
            'num_leaves': num_leaves,
            'bagging_fraction': bagging_fraction,
            'featur,e_fraction': feature_fraction,
#            'max_depth':-1,
#            'subsample_for_bin':200000,
#            'min_data_in_leaf': 50,
#             'lambda_l1': 0.5,
#             'lambda_l2': 0.5,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 3,
#             'bagging_fraction': 0.9,
            #'min_data_in_leaf': 50
            #'min_split_gain':0.0,
            #'bagging_fraction': 0.9, 'bagging_freq': 3,
            # 'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
            # 'lambda_l2': 0.5, 'min_data_in_leaf': 50
        }
        self.grow_boost_round = 10
        
        self.lr = get_log_lr(160, 0.23, 0.025) + [0.025]*3000
        self.step = 0
        self.models = [None]*class_num


    def fit(self, X_train, y_train, categorical_feature=None):
        
        start = time.time()
        
        class_num = y_train.shape[1]
        
        #self.models = [LGBMClassifier()]*class_num
        if not categorical_feature:
            categorical_feature = []
        
        num_boost_round = 230
        print ('###categorical_feature:', categorical_feature)
        lr = self.lr[0:num_boost_round]
        print ('###lr:', self.step, lr[:5])
        
        for i in range(class_num):
            y_tmp = y_train[:, i]
            
            #self.models[i] = LGBMClassifier()
            #self.models[i].fit(X_train, y_tmp)
            
            lgb_train = lgb.Dataset(X_train, label=y_tmp)
            print ('###model I:', i)
            self.models[i] = lgb.train(
                               params={**self.params, **self.hyperparams},
                               train_set=lgb_train, categorical_feature=categorical_feature,
                               #valid_sets=lgb_train, valid_names='train',
                               num_boost_round=num_boost_round,
                               learning_rates = lr,
                               # early_stopping_rounds=10,
                               verbose_eval=2,
                               )
        self.step += 1
        print ('###trainging time:', time.time() - start)


    def predict_proba(self, X_test):
        res = self.models[0].predict(X_test)
        #res = self.models[0].predict_proba(X_test)[:, 1]
        #print ('###res1:', res)
        for model in self.models[1:]:
            tmp = model.predict(X_test)
            #tmp = model.predict_proba(X_test)[:, 1]
            res = np.c_[res, tmp]
        #print ('###res2:', res)
        return res
    
    
class ClassifierChains:
    def __init__(self):
        self.model = ClassifierChain(LGBMClassifier())

    def set_grow_step(self, new_step):
        self.grow_boost_round = new_step

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test).A


def auc_metric(solution, prediction, task='binary.classification'):
    '''roc_auc_score() in sklearn is fast than code provided by sponsor
    '''
    if solution.sum(axis=0).min() == 0 :
        return np.nan
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc*2-1)


class SVM:
    def __init__(self, num_class):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
        self.num_class = num_class
        # self.model = CalibratedClassifierCV(LinearSVC(random_state=0, tol=1e-5, max_iter=300))
        self.model = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5, max_iter=100))

    def fit(self, X_train, y_train):
        X_train = self.imputer.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_train)
        print('pred: ', pred)
        # val_auc = auc_metric(np.eye(self.num_class)[y_train], pred)
        # print('metricx norm auc: ', val_auc)

    def predict(self, X_test):
        X_test = self.imputer.transform(X_test)
        # pred = self.model.predict_proba(X_test)
        pred = self.model.predict(X_test)
        return pred
        # return np.argmax(pred, axis=1)
