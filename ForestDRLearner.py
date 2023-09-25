#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from econml.orf import DMLOrthoForest
from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
np.random.seed(2023) 
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error 
import os
from joblib import dump, load 
from econml.orf import DMLOrthoForest 
from econml.dr import DRLearner, ForestDRLearner 
from sklearn.ensemble import GradientBoostingRegressor 


# In[ ]:





# In[113]:


def calc_score01(train_t1, test_data_t1):
    trn = train_t1.reshape(-1, 1)
    trn = pd.DataFrame(trn, columns=['effect1'])
    tst = pd.DataFrame({'effect1': test_data_t1}) 
    trn_tst = pd.concat([trn, tst], axis=0, ignore_index=True) 
    target = pd.read_csv('./dataset/data/target.csv')
    target = target.reset_index(drop=True)
    result = pd.concat([target, trn_tst], axis=1) 
    def calc_metric(result):
        r = np.sqrt(np.sum((result.ce_1 - result.effect1)**2)/result.shape[0])/result.ce_1.mean() 
        return r 
    return calc_metric(result) 

def calc_score02(train_t1, test_data_t1):
    trn = train_t1.reshape(-1, 1)
    trn = pd.DataFrame(trn, columns=['effect1'])
    tst = pd.DataFrame({'effect1': test_data_t1}) 
    trn_tst = pd.concat([trn, tst], axis=0, ignore_index=True) 
    target = pd.read_csv('./dataset/data/target.csv')
    target = target.reset_index(drop=True)
    result = pd.concat([target, trn_tst], axis=1) 
    def calc_metric(result):
        r = np.sqrt(np.sum((result.ce_2 - result.effect1)**2)/result.shape[0])/result.ce_2.mean() 
        return r 
    return calc_metric(result) 


def bayesGridSearchCVParams(X, Y, objective='regression', scoring=None):
    """  
    model: lgbm
    X, Y dtype: int, float, category
    objective:  'regression': 传统的均方误差回归。
                'regression_l1': 使用L1损失的回归，也称为 Mean Absolute Error (MAE)。
                'huber': 使用Huber损失的回归，这是均方误差和绝对误差的结合，特别适用于有异常值的情况。
                'fair': 使用Fair损失的回归，这也是另一种对异常值鲁棒的损失函数。
                'binary', 
                'multiclass'
    scoring:
    'neg_root_mean_squared_error', 'precision_micro', 'jaccard_micro', 'f1_macro', 
    'recall_weighted', 'neg_mean_absolute_percentage_error', 'f1_weighted', 
    'completeness_score', 'neg_brier_score', 'neg_mean_gamma_deviance', 'precision', 
    'adjusted_mutual_info_score', 'f1_samples', 'jaccard', 'neg_mean_poisson_deviance', 
    'precision_samples', 'recall', 'recall_samples', 'top_k_accuracy', 'roc_auc_ovr', 
    'mutual_info_score', 'jaccard_samples', 'positive_likelihood_ratio', 'f1_micro', 
    'adjusted_rand_score', 'accuracy', 'matthews_corrcoef', 'neg_mean_squared_log_error', 
    'precision_macro', 'rand_score', 'neg_log_loss', 'recall_macro', 'roc_auc_ovo', 
    'average_precision', 'jaccard_weighted', 'max_error', 'neg_median_absolute_error', 
    'jaccard_macro', 'roc_auc_ovo_weighted', 'fowlkes_mallows_score', 'precision_weighted', 
    'balanced_accuracy', 'v_measure_score', 'recall_micro', 'normalized_mutual_info_score', 
    'neg_mean_squared_error', 'roc_auc', 'roc_auc_ovr_weighted', 'f1', 'homogeneity_score', 
    'explained_variance', 'r2', 'neg_mean_absolute_error', 'neg_negative_likelihood_ratio'
    """ 
    if Y[Y.columns[0]].dtype in (np.float32, np.float64, np.int32, np.int64):
        y_type = 'regression' if objective is None else objective
    elif Y[Y.columns[0]].unique().shape[0]==2:
        y_type = 'binary'
    elif Y[Y.columns[0]].unique().shape[0] > 2:
        y_type = 'multiclass'
    else:
        raise ValueError('确认Y的类别数')
    print(y_type) 
    # grid 
    if y_type in ('multiclass', 'binary'): 
        params = {'boosting_type': 'gbdt', 'objective': y_type,
                'class_weight': 'balanced', 'n_jobs': -1} 
        estimator = lgb.LGBMClassifier(**params) 
        param_grid = {'learning_rate': Real(0.01, 0.03), 
                      'n_estimators': Integer(500, 2000),
                      'num_leaves': Integer(7, 31), 
                      'min_child_samples': Integer(3,  11), 
#                       'reg_alpha': Real(0.0, 0.1),
                      'reg_lambda': Real(0, 0.1), 
                      'seed': Categorical([42])} 
    else:
        params = {'boosting_type': 'gbdt',  'n_jobs': -1}
        estimator = lgb.LGBMRegressor(**params) 
        param_grid = {'objective': Categorical(['regression', 'regression_l1', 'huber', 'fair']),
                      'learning_rate': Real(0.01, 0.03), 
                      'n_estimators': Integer(1000, 4000),
                      'num_leaves': Integer(10, 20),
                      'min_child_samples': Integer(7, 20),
#                       'reg_alpha': 0,
#                       'reg_lambda': 0, 
                      'seed': Categorical([42])} 
    # search 
    grid = BayesSearchCV(estimator, param_grid, 
                         n_iter=300,
                         cv=3, scoring = scoring, n_jobs=-1, verbose=0)
    grid.fit(X, Y) 
    params.update(grid.best_params_)
    print('Best parameters found by grid search are:', params)
    return params 


# In[4]:


os.listdir('./dataset/data/best/')


# In[5]:


X = pd.read_csv('./dataset/data/best/X.csv', index_col=0)
X_01 = pd.read_csv('./dataset/data/best/X_01.csv', index_col=0)
X_02 = pd.read_csv('./dataset/data/best/X_02.csv', index_col=0)
test = pd.read_csv('./dataset/data/best/test.csv', index_col=0)

T = pd.read_csv('./dataset/data/best/T.csv')
Y = pd.read_csv('./dataset/data/best/Y.csv') 
T = T.astype(str).astype('category') 

X_T = pd.concat([X, T], axis=1) 
X_T_Y = pd.concat([X_T, Y], axis=1) 

T_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])][['T']]
Y_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])][['Y']]

T_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])][['T']]
Y_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])][['Y']]

T_02['T'] = T_02['T'].replace({'2': '1'}) 

X_T_01 = pd.concat([X_01, T_01], axis=1)
X_T_02 = pd.concat([X_02, T_02], axis=1) 

T_01['T'] = T_01['T'].astype(np.int64)
T_01 = np.array(T_01['T'])
T_02['T'] = T_02['T'].astype(np.int64)
T_02 = T_02.values.reshape(-1)

X_T_01['T'] = X_T_01['T'].astype(np.int64) 
X_Y_01 = pd.concat((X_01, Y_01), axis=1)


# In[60]:


X_01_BAL = X_01
T_01_BAL = T_01
Y_01_BAL = Y_01 
X_T_01_BAL = pd.concat((X_01_BAL.reset_index(drop=True), pd.DataFrame({"T":T_01_BAL})), axis=1)


# In[111]:


X_02_BAL = X_02
T_02_BAL = T_02
Y_02_BAL = Y_02 
X_T_02_BAL = pd.concat((X_02_BAL.reset_index(drop=True), pd.DataFrame({"T":T_02_BAL})), axis=1)


# In[ ]:





# In[80]:


# params_XT01 = bayesGridSearchCVParams(X_01_BAL, pd.DataFrame(T_01_BAL), objective='binary', scoring='roc_auc_ovr_weighted')
params_XT01 = {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.03, 'min_child_samples': 7, 'n_estimators': 500, 'num_leaves': 7, 'reg_lambda': 0.05, 'seed': 42}
mdl_t01 = lgb.LGBMClassifier(**params_XT01) 
# binary
# Best parameters found by grid search are: {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.03, 'min_child_samples': 7, 'n_estimators': 500, 'num_leaves': 7, 'reg_lambda': 0.05, 'seed': 42}


# In[81]:


# params_XTY01 = bayesGridSearchCVParams(X_T_01_BAL, Y_01_BAL, objective='regression', scoring='neg_root_mean_squared_error')
params_XTY01 = {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05, 'min_child_samples': 20, 'n_estimators': 3000, 'num_leaves': 7, 'objective': 'regression', 'reg_lambda': 0.01, 'seed': 42}
mdl_y01 = lgb.LGBMRegressor(**params_XTY01) 
# regression
# Best parameters found by grid search are: {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05, 'min_child_samples': 20, 'n_estimators': 3000, 'num_leaves': 7, 'objective': 'regression', 'reg_lambda': 0.01, 'seed': 42}


# In[ ]:





# In[87]:


n_estimators_list = [500]# fitted
min_leaf_size_list = [60]# 20-90
subsample_ratio_list = [0.35]#fiteed
max_depth_list = [8]#fitted 
max_features_list = [0.9]# fitted
min_balancedness_tol_list = [0.45]#fitted, 0.4, 0.35, 0.3]
for n_est in n_estimators_list: 
    for min_leaf in min_leaf_size_list:
        for subsample_ratio in subsample_ratio_list:
            for max_dep in max_depth_list:
                for min_balance in min_balancedness_tol_list:
                    for max_feat in max_features_list:
                        params_tmp = {'n_estimators': n_est, 
                                   'min_leaf_size': min_leaf, 
                                   'max_depth': max_dep, 
                                   'subsample_ratio': subsample_ratio, 
                                   'min_balance': min_balance, 
                                   'max_feat': max_feat} 
                    # 01 
                        est01 = ForestDRLearner(
                                        model_regression=mdl_y01, 
                                        model_propensity=mdl_t01, 
                                        featurizer=None, 
                                        min_propensity=1e-06, 
                                        categories=[0, 1], 
                                        cv=2,
                                        mc_iters=4,
                                        mc_agg='mean', 
                                        n_estimators=n_est, 
                                        max_depth=max_dep, 
#                                         min_samples_split=5, 
                                        min_samples_leaf=min_leaf, 
#                                         min_weight_fraction_leaf=0.0, 
                                        max_features=max_feat, 
#                                         min_impurity_decrease=0.0, 
                                        max_samples=subsample_ratio, 
                                        min_balancedness_tol=min_balance, 
                                        honest=True, 
                                        subforest_size=4, 
                                        n_jobs = -1, 
                                        verbose=0, 
                                        random_state=0)
                    
                        # fit
                        est01.fit(Y=Y_01_BAL, T=T_01_BAL, X=X_01_BAL)
                        # effect
                        test_t01 = est01.effect(X=test, T0=0, T1=1) 
                        X_t01 = est01.effect(X=X, T0=0, T1=1)
                        print(params_tmp, calc_score(X_t01.reshape(-1), test_t01.reshape(-1)))


# In[ ]:


# {'n_estimators': 500, 'min_leaf_size': 80, 'max_depth': 7, 'subsample_ratio': 0.35, 
#  'min_balance': 0.45, 'max_feat': 0.9} 0.37243761942135306


# In[93]:


# pd.DataFrame(X_t01,columns=['X_t01']).to_csv('./dataset/model/forestDRlearner/X_t01.csv',index=False) 
# pd.DataFrame(test_t01,columns=['test_t01']).to_csv('./dataset/model/forestDRlearner/test_t01.csv',index=False)  


# In[ ]:


# 0.36  {'n_estimators': 500, 'min_leaf_size': 60, 'max_depth': 8, 'subsample_ratio': 0.35, 'min_balance': 0.45, 'max_feat': 0.9} 0.36057932254417746


# In[96]:


# from joblib import dump
# dump(est01, './dataset/model/forestDRlearner/est01.joblib')


# In[ ]:





# ##### est02

# In[114]:


params_XT02 = bayesGridSearchCVParams(X_02_BAL, pd.DataFrame(T_02_BAL), 
                                    objective='binary', scoring='roc_auc_ovr_weighted')
mdl_t02 = lgb.LGBMClassifier(**params_XT02) 


# In[115]:


params_XTY02 = bayesGridSearchCVParams(X_T_02_BAL, Y_02_BAL, 
                                     objective='regression', scoring='neg_root_mean_squared_error')
mdl_y02 = lgb.LGBMRegressor(**params_XTY02) 


# In[137]:


n_estimators_list = [300]# fitted
min_leaf_size_list = [20]# 20-90?
subsample_ratio_list = [0.33]#fiteed
max_depth_list = [12]#fitted 
max_features_list = [0.9]# fitted
min_balancedness_tol_list = [0.45]#fitted, 0.4, 0.35, 0.3]
for n_est in n_estimators_list: 
    for min_leaf in min_leaf_size_list:
        for subsample_ratio in subsample_ratio_list:
            for max_dep in max_depth_list:
                for min_balance in min_balancedness_tol_list:
                    for max_feat in max_features_list:
                        params_tmp = {'n_estimators': n_est, 
                                   'min_leaf_size': min_leaf, 
                                   'max_depth': max_dep, 
                                   'subsample_ratio': subsample_ratio, 
                                   'min_balance': min_balance, 
                                   'max_feat': max_feat} 
                    # 01 
                        est02 = ForestDRLearner(
                                        model_regression=mdl_y02, 
                                        model_propensity=mdl_t02, 
                                        featurizer=None, 
                                        min_propensity=1e-06, 
                                        categories=[0, 1], 
                                        cv=2,
                                        mc_iters=5,
                                        mc_agg='mean', 
                                        n_estimators=n_est, 
                                        max_depth=max_dep, 
#                                         min_samples_split=5, 
                                        min_samples_leaf=min_leaf, 
#                                         min_weight_fraction_leaf=0.0, 
                                        max_features=max_feat, 
#                                         min_impurity_decrease=0.0, 
                                        max_samples=subsample_ratio, 
                                        min_balancedness_tol=min_balance, 
                                        honest=True, 
                                        subforest_size=4, 
                                        n_jobs = -1, 
                                        verbose=0, 
                                        random_state=0)
                    
                        # fit
                        est02.fit(Y=Y_02_BAL, T=T_02_BAL, X=X_02_BAL)
                        # effect
                        test_t02 = est02.effect(X=test, T0=0, T1=1) 
                        X_t02 = est02.effect(X=X, T0=0, T1=1)
                        print(params_tmp, calc_score02(X_t02.reshape(-1), test_t02.reshape(-1)))


# In[138]:


# pd.DataFrame(X_t02, columns=['X_t02']).to_csv('./dataset/model/forestDRlearner/X_t02.csv',index=False) 
# pd.DataFrame(test_t02, columns=['test_t02']).to_csv('./dataset/model/forestDRlearner/test_t02.csv',index=False) 


# In[139]:


# from joblib import dump
# dump(est02, './dataset/model/forestDRlearner/est02.joblib')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




