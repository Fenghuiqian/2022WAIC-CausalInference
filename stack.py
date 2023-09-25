#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.model_selection import KFold
from itertools import combinations
import os


# In[2]:


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
#     print(y_type) 
    # grid 
    if y_type in ('multiclass', 'binary'): 
        params = {'boosting_type': 'gbdt', 'objective': y_type,
                'class_weight': 'balanced', 'n_jobs': -1} 
        estimator = lgb.LGBMClassifier(**params) 
        param_grid = {'learning_rate': [0.01, 0.03, 0.05], 
                      'n_estimators': [500, 1000, 2000, 3000],
                      'num_leaves': [7, 15, 31, 63],
                      'min_child_samples': [1, 3, 5, 7 , 11], 
#                       'reg_alpha': [0, 0.01, 0.05, 0.1],
                      'reg_lambda': [0, 0.01, 0.05, 0.1], 
                      'seed': [42]} 
    else:
        params = {'boosting_type': 'gbdt',  'n_jobs': -1}
        estimator = lgb.LGBMRegressor(**params) 
        param_grid = {
                      'objective': ['regression'], # 'objective': ['regression', 'regression_l1', 'huber', 'fair'],
                      'learning_rate': [0.03],  # 'learning_rate': [0.03], 0.01, 0.03, 0.05], 
                      'n_estimators': [200, 300, 400, 500, 800],  #'n_estimators': [300, 500, 1000, 2000, 3000], 
                      'num_leaves': [1, 3, 5, 7, 9],      # 'num_leaves': [7, 15, 31, 63, 127],
                      'min_child_samples': [14, 18, 22, 25],  # 'min_child_samples': [1, 3, 5, 7, 10, 15, 20, 30],
                      'reg_alpha': [0,  0.02, 0.05],
                      'reg_lambda': [0,  0.02, 0.05], 
                      'seed': [42], 
                      'n_jobs': [-1]} 
    # search 
#     print(scoring) 
    grid = GridSearchCV(estimator, param_grid, 
#                          n_iter=300,
                         cv=3, scoring = scoring, n_jobs=-1, verbose=0)
    grid.fit(X, Y) 
    params.update(grid.best_params_)
#     print('Best parameters found by grid search are:', params)
#     print('best Score:', grid.best_score_)
    return params 

def kfold_pred(X_train, X_test, y_train, y_test):
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    for train_index, valid_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]
        tmp_params = bayesGridSearchCVParams(X_tr, y_tr, objective='regression', scoring='neg_root_mean_squared_error')
        model = lgb.LGBMRegressor(**tmp_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                            early_stopping_rounds=50, verbose=False) 
        # Predict out-of-fold part of the training data
        # as predict, 不能多次预测降低方差
#         oof_preds[valid_index] = model.predict(X_val, num_iteration=model.best_iteration_)
        # Average test predictions over the folds
    # 多次预测取平均
        test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / kf.n_splits
#     mse_oof = mean_squared_error(y_train, oof_preds)
#     print(f"Mean Squared Error for OOF predictions: {mse_oof:.4f}")
#     rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
    print(test_preds.shape)
    return test_preds 


# In[ ]:





# ## 结果

# In[3]:


cf_T01 = pd.read_csv('./dataset/model/cfdml/cft01.csv')
cf_T01 = cf_T01.values

fdr_xt01 = pd.read_csv('./dataset/model/forestDRlearner/X_t01.csv')
fdr_testt01 = pd.read_csv('./dataset/model/forestDRlearner/test_t01.csv')
fdr_T01 = np.concatenate((fdr_xt01.values, fdr_testt01.values), axis=0) 

dr1_T01 = pd.read_csv('./dataset/model/drlearner/drt01.csv')
dr1_T01 = dr1_T01.values

dr2_T01 = pd.read_csv('./dataset/model/drlearner2/drt01.csv')
dr2_T01 = dr2_T01.values


# In[48]:


t01 = pd.DataFrame({'cf_T01': cf_T01.reshape(-1), 'fdr_T01': fdr_T01.reshape(-1), 
                   'dr1_T01': dr1_T01.reshape(-1), 'dr2_T01': dr2_T01.reshape(-1)}) 


# In[50]:


cols1 = t01.columns.to_list()
t01['max_'] = np.max(t01[cols1], axis=1)
t01['min_'] = np.min(t01[cols1], axis=1)
t01['mean_'] = np.mean(t01[cols1], axis=1)
t01['std_'] = np.std(t01[cols1], axis=1) 
t01['delta_'] = t01.max_ - t01.min_ 
iter1 = combinations(cols1, 2)
for a, b in iter1:
    t01['delta_%s'%a[:-4]+b[:-4]] = t01[a] - t01[b] 


# In[51]:


target1 = pd.read_csv('./dataset/data/target.csv')
target1 = target1[['ce_1']]


# In[53]:


kf0 = KFold(n_splits=4, shuffle=True, random_state=0)
result_pred1 = np.zeros(t01.shape[0])
for tr_idx, tst_idx in kf0.split(t01):
    X_train, X_test = t01.iloc[tr_idx], t01.iloc[tst_idx]
    y_train, y_test = target1.iloc[tr_idx], target1.iloc[tst_idx]
    pred_ = kfold_pred(X_train, X_test, y_train, y_test)
    result_pred1[tst_idx] = pred_
    print('nan Count', result_pred1[result_pred1==0].shape) 
    


# In[54]:


calc_score01(result_pred1[:-5000], result_pred1[-5000:]) 


# In[ ]:





# In[ ]:





# ##### t02

# In[55]:


cf_T02 = pd.read_csv('./dataset/model/cfdml/cft02.csv')
cf_T02 = cf_T02.values

fdr_xt02 = pd.read_csv('./dataset/model/forestDRlearner/X_t02.csv')
fdr_testt02 = pd.read_csv('./dataset/model/forestDRlearner/test_t02.csv')
fdr_T02 = np.concatenate((fdr_xt02.values, fdr_testt02.values), axis=0) 

dr1_T02 = pd.read_csv('./dataset/model/drlearner/drt02.csv')
dr1_T02 = dr1_T02.values

dr2_T02 = pd.read_csv('./dataset/model/drlearner2/drt02.csv')
dr2_T02 = dr2_T02.values


# In[56]:


t02 = pd.DataFrame({'cf_T02': cf_T02.reshape(-1), 'fdr_T02': fdr_T02.reshape(-1), 
                   'dr1_T02': dr1_T02.reshape(-1), 'dr2_T02': dr2_T02.reshape(-1)}) 


# In[57]:


cols2 = t02.columns.to_list()
t02['max_'] = np.max(t02[cols2], axis=1)
t02['min_'] = np.min(t02[cols2], axis=1)
t02['mean_'] = np.mean(t02[cols2], axis=1)
t02['std_'] = np.std(t02[cols2], axis=1) 
t02['delta_'] = t02.max_ - t02.min_ 
iter2 = combinations(cols2, 2)
for a, b in iter2:
    t02['delta_%s'%a[:-4]+b[:-4]] = t02[a] - t02[b] 


# In[ ]:





# In[59]:


target2 = pd.read_csv('./dataset/data/target.csv')
target2 = target2[['ce_2']]


# In[60]:


kf2 = KFold(n_splits=4, shuffle=True, random_state=0)
result_pred2 = np.zeros(t02.shape[0])
for tr_idx, tst_idx in kf2.split(t02):
    X_train, X_test = t02.iloc[tr_idx], t02.iloc[tst_idx]
    y_train, y_test = target2.iloc[tr_idx], target2.iloc[tst_idx]
    pred_ = kfold_pred(X_train, X_test, y_train, y_test)
    result_pred2[tst_idx] = pred_
    print('nan Count', result_pred2[result_pred2==0].shape) 


# In[61]:


calc_score02(result_pred2[:-5000], result_pred2[-5000:]) 


# In[62]:


def calc_score(pred01, pred02):
    result = pd.DataFrame(np.concatenate((pred01.reshape(-1, 1), pred02.reshape(-1, 1)), axis=1), 
                    columns=['pred01', 'pred02'])
    target = pd.read_csv('./dataset/data/target.csv')
    result = pd.concat([target, result], axis=1)
    def calc_metric(result):
        r = np.sqrt(np.sum((result.ce_1 - result.pred01)**2)/result.shape[0])/result.ce_1.mean() +             np.sqrt(np.sum((result.ce_2 - result.pred02)**2)/result.shape[0])/result.ce_2.mean()
        return r 
    return calc_metric(result) 


# In[63]:


calc_score(result_pred1, result_pred2)


# In[64]:


pd.DataFrame(np.concatenate((result_pred1.reshape(-1, 1), result_pred2.reshape(-1, 1)), 
            axis=1), columns=['pred01', 'pred02']).to_csv('./dataset/stacked/result.csv')


# In[ ]:




