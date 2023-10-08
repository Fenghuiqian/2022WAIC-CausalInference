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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error 
from tqdm import tqdm
import skopt
from econml.dml import CausalForestDML 
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge 


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


# In[3]:


def bayesGridSearchCVParams_Lgbm(X, Y, cv=4, n_iter=100, objective='regression', scoring=None):
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
                'class_weight': 'balanced', 'n_jobs': -1, 'seed': 42} 
        estimator = lgb.LGBMClassifier(**params)
        param_grid = {'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(100, 3000),
                      'num_leaves': Integer(3, 31),
                      'max_depth':Integer(2, 10), 
                      'min_child_samples': Integer(1,  30), 
                      'min_data_in_bin': Categorical([1, 2, 3, 5]),
                      'reg_alpha': Real(0, 0.2),
                      'reg_lambda': Real(0, 0.2)
                     } 
    
    else:
        params = {'boosting_type': 'gbdt',  'n_jobs': -1, 'seed': 42}
        estimator = lgb.LGBMRegressor(**params) 
        param_grid = {'objective': Categorical(['regression', 'regression_l1', 'huber', 'fair']),
                      'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(100, 3000),
                      'num_leaves': Integer(3, 31),
                      'max_depth': Integer(2, 10), 
                      'min_child_samples': Integer(1, 30),
                      'min_data_in_bin': Categorical([1, 2, 3, 5]),
                      'reg_alpha': Real(0, 0.2),
                      'reg_lambda': Real(0, 0.2)}

    # search 
    if scoring is None:
        scoring = 'neg_root_mean_squared_error' if y_type == 'regression' else 'roc_auc_ovr_weighted' 
    print(scoring) 
    grid = BayesSearchCV(estimator, param_grid, 
                         n_iter=n_iter,
                         cv=cv, scoring = scoring, n_jobs=-1, verbose=1)
    grid.fit(X, Y) 
    params.update(grid.best_params_)
    print('Best parameters found by grid search are:', params)
    print('best score', grid.best_score_)
    return params 


# ##### data

# In[4]:


X = pd.read_csv('./dataset/data/best/X.csv',index_col=0)
test = pd.read_csv('./dataset/data/best/test.csv', index_col=0)
test = test[X.columns]
T = pd.read_csv('./dataset/data/best/T.csv')
Y = pd.read_csv('./dataset/data/best/Y.csv') 
T = T.astype(str).astype('category') 


# In[5]:


X_T = pd.concat([X, T], axis=1) 
X_T_Y = pd.concat([X_T, Y], axis=1) 
X_T.shape, X_T_Y.shape


# In[9]:


T_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])][['T']]
Y_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])][['Y']]
X_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])].drop(columns=['T', 'Y']) 
X_T_01 = X_T_Y.loc[X_T_Y['T'].isin(['0', '1'])].drop(columns=['Y']) 
T_01['T'] = T_01['T'].astype(str).astype('category')
X_T_01['T'] = X_T_01['T'].astype(str).astype('category')
Y_01.shape, X_01.shape, T_01.shape 


# In[10]:


pc_data = X_T_Y.astype(np.float64).values


# In[11]:


cg = pc(pc_data, alpha = 0.15, indep_test = fisherz, 
        stable = True, uc_rule = 1, uc_priority = 1) 


# In[12]:


cg.draw_pydot_graph(labels = X_T_Y.columns.tolist()) 


# In[ ]:


# W是与T和Y都有关系的混淆变量
# 图中可能的W：['V_2', 'V_15'] 


# In[ ]:





# ##### residual model

# In[13]:


# params_XT01 = bayesGridSearchCVParams_Lgbm(X_01, T_01, cv=3, n_iter=100, objective='binary', 
#                                            scoring='roc_auc_ovr_weighted') 
params_XT01 = {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.03, 'min_child_samples': 7, 'n_estimators': 500, 'num_leaves': 7, 'reg_lambda': 0.05, 'seed': 42}
mdl_t01 = lgb.LGBMClassifier(**params_XT01) 


# In[14]:


# params_XY01 = bayesGridSearchCVParams(X_01, Y_01, objective='regression', scoring='neg_mean_absolute_error') 
params_XY01 = {'boosting_type': 'gbdt', 'objective': 'regression',   'n_jobs': -1,  'learning_rate': 0.03, 'min_child_samples': 4, 'n_estimators': 1000,   'num_leaves': 15,  'seed': 42}
mdl_y01 = lgb.LGBMRegressor(**params_XY01) 


# In[ ]:





# ### est01

# In[ ]:


X_01.columns


# In[31]:


W = ['V_8_no', 'V_31',  'V_9']


# In[32]:


max_features_list = [0.9]
n_estimators_list = [500]
max_depth_list = [8]
min_samples_split_list = [200] 
max_samples_list = [0.28]
for max_fea in max_features_list:
    for n_est in n_estimators_list:
        for max_dep in max_depth_list:
            for min_samp in min_samples_split_list:
                for max_samp in max_samples_list:
                    para_tmp = {'n_estimators': n_est, 
                               'min_samples_split': min_samp, 
                               'max_samples': max_samp, 
                               'max_features': max_fea, 
                               'max_depth': max_dep} 
                    est01 = CausalForestDML( 
                                    model_y = mdl_y01, model_t = mdl_t01,
                                    discrete_treatment =True, 
                                    categories =['0', '1'], 
                                    mc_iters =4, mc_agg = 'mean', 
                                    drate =True,
                                    criterion ='mse', 
                                    featurizer = None, 
                                    max_depth = max_dep, 
                                    n_estimators = n_est, 
                                    min_samples_split = min_samp, 
                                    max_samples = max_samp, 
                                    honest=True, 
#                                     min_weight_fraction_leaf = 0.01 ,
#                                     min_var_leaf_on_val = False , 
                                    inference =True,
                                    max_features = max_fea,
                                    n_jobs =-1, 
                                    random_state =2023,
                                    verbose =0 ) 
                    est01.fit(Y=Y_01, T=T_01, X=X_01.drop(columns=W), W=X_01[W]) 
                    test_t01 = est01.effect(X=test.drop(columns=W), T0='0', T1='1') 
                    X_t01 = est01.effect(X=X.drop(columns=W), T0='0', T1='1')  
                    print(para_tmp, calc_score01(X_t01.reshape(-1), test_t01.reshape(-1)))


# In[30]:


# W=[], 0.314086464339124
# W = ['V_15']  0.3220646206261919
# W = ['V_2'] 0.31439440193893814 
# W = ['V_2', 'V_15'] 0.3158790264348197 
# W = ['V_2', 'V_15', 'V_8_no']  0.3111122470174761
# W = ['V_8_no']   0.3113633836369264
# W = ['V_8_no', 'V_31']   0.3058371154389804
# W = ['V_31'] 0.3134274610773896 
# W = ['V_8_no', 'V_31', 'V_39', 'V_10_no']  0.3106958438985337
# W = ['V_8_no', 'V_31',  'V_9']   0.305272262259072
# 结论： 'V_8_no', 'V_31',  'V_9'可能是W 


# In[ ]:





# In[ ]:


# from joblib import dump
# dump(est01, './dataset/model/causalForest/causalForest0_31_MODEL_est01.joblib')
# pd.DataFrame(test_t01, columns=['test_t01']).to_csv('./dataset/model/causalForest/test_t01.csv', index=False) 
# pd.DataFrame(X_t01, columns=['X_t01']).to_csv('./dataset/model/causalForest/X_t01.csv', index=False) 


# In[ ]:





# In[ ]:





# ### est02

# In[33]:


T_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])][['T']]
Y_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])][['Y']]
X_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])].drop(columns=['T', 'Y']) 
X_T_02 = X_T_Y.loc[X_T_Y['T'].isin(['0', '2'])].drop(columns=['Y']) 
T_02['T'] = T_02['T'].astype(str).astype('category')
X_T_02['T'] = X_T_02['T'].astype(str).astype('category')
Y_02.shape, X_02.shape, T_02.shape


# In[34]:


# params_XT02 = bayesGridSearchCVParams(X_02, T_02, objective='binary', scoring='roc_auc_ovr_weighted') 
params_XT02 = {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.010618497852749719, 'min_child_samples': 11, 'n_estimators': 500, 'num_leaves': 7, 'reg_lambda': 0.1, 'seed': 42}
mdl_t02 = lgb.LGBMClassifier(**params_XT02) 


# In[35]:


# params_XY02 = bayesGridSearchCVParams(X_02, Y_02, objective='regression', scoring='neg_mean_absolute_error') 
params_XY02 = {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.03, 'min_child_samples': 15, 'n_estimators': 4000, 'num_leaves': 10, 'objective': 'regression_l1', 'reg_lambda': 0, 'seed': 42}
mdl_y02 = lgb.LGBMRegressor(**params_XY02) 


# In[36]:


X_02.columns


# In[37]:


W = ['V_8_no', 'V_31', 'V_39', 'V_10_no']    


# In[38]:


max_samples_list = [0.5]
max_features_list = [0.9]
min_samples_split_list = [200]
n_estimators_list = [500]
max_depth_list = [8] 
for max_fea in max_features_list:
    for max_samp in max_samples_list:
        for min_samp in min_samples_split_list:
            for n_est in n_estimators_list:
                for max_dep in max_depth_list:
                    params_tmp = {'n_estimators': n_est, 
                               'min_samples_split': min_samp, 
                               'max_samples': max_samp, 
                               'max_features': max_fea, 
                               'max_depth': max_dep} 

                    est02 = CausalForestDML( 
                                    model_y = mdl_y02, model_t = mdl_t02,
                                    discrete_treatment =True, 
                                    categories =['0', '2'], 
                                    mc_iters =4, mc_agg = 'mean', 
                                    drate = True,
                                    criterion ='het', 
                                    featurizer = None,
                                    max_depth = max_dep, 
                                    n_estimators = n_est, 
                                    min_samples_split = min_samp, 
                                    max_samples = max_samp,
                                    honest=True, 
#                                     min_weight_fraction_leaf = 0.01 ,
#                                     min_var_leaf_on_val = False , 
                                    inference =True,
                                    max_features = max_fea,
                                    n_jobs =-1, 
                                    random_state =2023,
                                    verbose =0 ) 
                    est02.fit(Y=Y_02, T=T_02, X=X_02.drop(columns=W), W=X_02[W])
                    test_t02 = est02.effect(X=test.drop(columns=W), T0='0', T1='2') 
                    X_t02 = est02.effect(X=X.drop(columns=W), T0='0', T1='2') 
                    print(params_tmp, calc_score02(X_t02.reshape(-1), test_t02.reshape(-1))) 


# In[39]:


# W = [], 0.04965519571874911
# ['V_8_no', 'V_31',  'V_9'] 0.14530027676500637 
# 'V_8_no' 0.049427037589570975  是W
# V_31   0.05032426511012993 
#  V_9   0.1334085770583134
# W = ['V_8_no', 'V_31']   0.04883403274399613 
# W = ['V_8_no', 'V_31', 'V_39', 'V_10_no']   0.04789656181025698
# 结论：['V_8_no', 'V_31', 'V_39', 'V_10_no'] 可能是W 


# In[ ]:





# ##### 结论

# In[40]:


# T01测试的结论： 'V_8_no', 'V_31',  'V_9'可能是W 
# T02测试的结论：['V_8_no', 'V_31', 'V_39', 'V_10_no'] 可能是W  
# 因此，['V_8_no', 'V_31']是混淆变量的可能性比较大。


# In[ ]:




