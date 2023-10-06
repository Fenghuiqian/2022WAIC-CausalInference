#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from econml.sklearn_extensions.linear_model import WeightedLasso
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
np.random.seed(2023) 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
from joblib import dump, load 
from econml.orf import DMLOrthoForest 
from econml.dr import DRLearner, ForestDRLearner 
from sklearn.ensemble import GradientBoostingRegressor 


# In[ ]:





# In[2]:


def calc_score01(train_t1, test_data_t1):
    trn = train_t1.reshape(-1, 1)
    trn = pd.DataFrame(trn, columns=['effect1'])
    tst = pd.DataFrame({'effect1': test_data_t1.reshape(-1)}) 
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
    tst = pd.DataFrame({'effect1': test_data_t1.reshape(-1)}) 
    trn_tst = pd.concat([trn, tst], axis=0, ignore_index=True) 
    target = pd.read_csv('./dataset/data/target.csv')
    target = target.reset_index(drop=True)
    result = pd.concat([target, trn_tst], axis=1) 
    def calc_metric(result):
        r = np.sqrt(np.sum((result.ce_2 - result.effect1)**2)/result.shape[0])/result.ce_2.mean() 
        return r 
    return calc_metric(result) 

def calc_score(pred01, pred02):
    result = pd.DataFrame(np.concatenate((pred01.reshape(-1, 1), pred02.reshape(-1, 1)), axis=1), 
                    columns=['pred01', 'pred02'])
    target = pd.read_csv('./dataset/data/target.csv')
    result = pd.concat([target, result], axis=1)
    def calc_metric(result):
        r = np.sqrt(np.sum((result.ce_1 - result.pred01)**2)/result.shape[0])/result.ce_1.mean() +             np.sqrt(np.sum((result.ce_2 - result.pred02)**2)/result.shape[0])/result.ce_2.mean()
        return r 
    return calc_metric(result) 
def bayesGridSearchCVParams(X, Y, objective='regression'):
    """ 
    X, Y dtype: int, float, category
    objective:  regression: 传统的均方误差回归。
                regression_l1: 使用L1损失的回归，也称为 Mean Absolute Error (MAE)。
                huber: 使用Huber损失的回归，这是均方误差和绝对误差的结合，特别适用于有异常值的情况。
                fair: 使用Fair损失的回归，这也是另一种对异常值鲁棒的损失函数。

                'binary', 
                'multiclass'
    model: lgbm
    """ 
    if Y[Y.columns[0]].dtype in (float, int):
        y_type = 'regression'
    elif Y[Y.columns[0]].unique().shape[0]==2:
        y_type = 'binary'
    elif Y[Y.columns[0]].unique().shape[0] > 2:
        y_type = 'multiclass'
    else:
        raise ValueError('确认Y的类别数')
    print(y_type) 
    if objective != y_type:
        raise ValueError('确认Y的类型')
    # grid 
    if y_type in ('multiclass', 'binary'): 
        params = {'boosting_type': 'gbdt', 'objective': y_type,
                'class_weight': 'balanced', 'n_jobs': -1} 
        estimator = lgb.LGBMClassifier(**params) 
        param_grid = {'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(200, 2000),
                      'num_leaves': Integer(7, 255), 
                      'min_child_samples': Integer(1, 11), 
                      'reg_alpha': Real(0.0, 0.1),
                      'reg_lambda': Real(0, 0.1), 
                      'seed': Categorical([42])} 
        scoring = 'roc_auc'
    else:
        params = {'boosting_type': 'gbdt',  'n_jobs': -1}
        estimator = lgb.LGBMRegressor(**params) 
        param_grid = {'objective': Categorical(['regression', 'regression_l1', 'huber', 'fair']),
                      'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(200, 2000),
                      'num_leaves': Integer(7, 255),  
                      'min_child_samples':Integer(1, 20),
                      'reg_alpha': Real(0.0, 0.1),
                      'reg_lambda': Real(0, 0.1), 
                      'seed': Categorical([42])} 
        scoring = 'neg_root_mean_squared_error' # 或 'r2'
    # search 
    grid = BayesSearchCV(estimator, param_grid, 
                         n_iter=300,
                         cv=4, scoring = scoring, n_jobs=-1)
    grid.fit(X, Y) 
    params.update(grid.best_params_)
    print('Best parameters found by grid search are:', params)
    return params 


# In[3]:


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


# In[4]:


T_01['T'] = T_01['T'].astype(str).astype('category')
T_02['T'] = T_02['T'].astype(str).astype('category') 


# In[20]:


# params_XT01 = bayesGridSearchCVParams(X_01, T_01, objective='binary')
params_XT01 = {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.05, 'min_child_samples': 11,  'n_estimators': 600, 'num_leaves': 7,  'reg_alpha': 0.01,  'reg_lambda': 0.1, 'seed': 42} 
mdl_t01 = lgb.LGBMClassifier(**params_XT01) 


# In[21]:


params_XTY01 = {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05, 'n_estimators': 1500, 'num_leaves': 15, 'objective': 'regression', 'reg_alpha': 0.05, 'reg_lambda': 0.05, 'seed': 42}
mdl_y01 = lgb.LGBMRegressor(**params_XTY01) 


# In[ ]:





# In[22]:


# params_XTY02 = bayesGridSearchCVParams(X_T_02, Y_02, objective='regression') # 
params_XTY02 = {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05,  'n_estimators': 2000, 'num_leaves': 15, 'objective': 'regression', 
                'reg_lambda': 0.01, 'reg_alpha': 0.05, 'seed': 42} 
mdl_y02 = lgb.LGBMRegressor(**params_XTY02)  


# In[ ]:





# In[23]:


# params_XT02 = bayesGridSearchCVParams(X_02, pd.DataFrame(T_02),  objective='binary')
params_XT02 = {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.03, 
               'min_child_samples': 3, 'n_estimators': 200, 'num_leaves': 7, 'reg_lambda': 0.01, 'seed': 42}
mdl_t02 = lgb.LGBMClassifier(**params_XT02) 


# In[ ]:





# In[24]:


cv_list = [3]
mc_iters_list = [3]
alpha_list = [0.6]
for cv in cv_list: 
    for mc_iters in mc_iters_list:
        for alpha in alpha_list:
            params_tmp = {'cv': cv, 'mc_iters': mc_iters, 'alpha': alpha} 
            # 01 
            est01 = DRLearner(  model_propensity=mdl_t01, 
                                model_regression=mdl_y01, 
                                model_final=WeightedLasso(
                                                        alpha=alpha,
                                                        random_state=0), 
                                multitask_model_final=False, 
                                featurizer=None, 
                                min_propensity=1e-06, 
                                categories=['0', '1'], 
                                cv=cv, 
                                mc_iters=mc_iters, 
                                mc_agg='mean', 
                                random_state=2023) 
            est01.fit(Y=Y_01, T=T_01, X=X_01) 
            test_t01 = est01.effect(X=test, T0='0', T1='1') 
            X_t01 = est01.effect(X=X, T0='0', T1='1') 
            score_ = calc_score01(X_t01, test_t01)
            print(params_tmp, score_)


# In[ ]:





# ##### save model/data

# In[ ]:


# from joblib import dump
# dump(est01, './dataset/model/DR0_57_MODEL_est01.joblib')
# dump(est02, './dataset/model/DR0_57_MODEL_est02.joblib')
# pd.DataFrame(test_t02, columns=['test_t02']).to_csv(
#     './dataset/model/DRMODEL0_57/test_t02.csv', index=False) 


# In[31]:


cv_list = [2]
mc_iters_list = [2]
alpha_list = [0.15]
for cv in cv_list: 
    for mc_iters in mc_iters_list:
        for alpha in alpha_list:
            params_tmp = {'cv': cv, 'mc_iters': mc_iters, 'alpha': alpha} 
            # 02
            est02 = DRLearner(model_propensity=mdl_t02, 
                                       model_regression=mdl_y02, 
                                       model_final=WeightedLasso(
                                                           alpha=alpha, 
                                                           random_state=0), 
                                        multitask_model_final=False, 
                                        featurizer=None, 
                                        min_propensity=1e-06, 
                                        categories=['0', '1'], 
                                        cv=cv,
                                        mc_iters=mc_iters,
                                        mc_agg='mean', 
                                        random_state=2023)

            est02.fit(Y=Y_02, T=T_02, X=X_02)
            test_t02 = est02.effect(X=test, T0='0', T1='1') 
            X_t02 = est02.effect(X=X, T0='0', T1='1')
            score_ = calc_score02(X_t02, test_t02) 
            print(params_tmp, score_)


# In[32]:


# 分数


# In[33]:


calc_score(np.concatenate((X_t01, test_t01), axis=0), np.concatenate((X_t02, test_t02), axis=0))


# In[ ]:





# In[34]:


pd.DataFrame(np.concatenate((X_t01, test_t01), axis=0), columns=['t01']).to_csv('./dataset/model/drlearner2/drt01.csv', index=False)

pd.DataFrame(np.concatenate((X_t02, test_t02), axis=0), columns=['t02']).to_csv('./dataset/model/drlearner2/drt02.csv', index=False)


# In[ ]:





# In[35]:


from joblib import dump
dump(est01, './dataset/model/drlearner2/est01.joblib')
dump(est02, './dataset/model/drlearner2/est02.joblib')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




