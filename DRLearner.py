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


def optunaSearchCVParams_LGBM(X, Y, cv=6, n_iter=30, sampler='tpe',  study_name = 'new',
                              objective_type='binary',  scoring='average_precision', direction='maximize',
                             n_jobs_est=10, n_jobs_optuna=3, use_gpu=False):
    
    """
    direction:  'minimize', 
                'maximize'
    objective_type: binary, multiclass, 
    
    scoring: binary: 'average_precision','roc_auc'
             multiclass: 'log_loss', 'accuracy_score'
             
             
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

             
    optuna.samplers.GridSampler(网格搜索采样)
    optuna.samplers.RandomSampler(随机搜索采样)
    optuna.samplers.TPESampler(贝叶斯优化采样)
    optuna.samplers.NSGAIISampler(遗传算法采样)
    optuna.samplers.CmaEsSampler(协方差矩阵自适应演化策略采样，非常先进的优化算法)

    贝叶斯优化（TPESampler）:
    基于过去的试验结果来选择新的参数组合，通常可以更快地找到好的解。
    当参数间的依赖关系比较复杂时，可能会更有优势。
    
    遗传算法（NSGAIISampler）:
    遗传算法是一种启发式的搜索方法，通过模拟自然选择过程来探索参数空间。
    对于非凸或非线性问题可能表现良好。
    
    协方差矩阵自适应演化策略（CmaEsSampler）:
    CMA-ES是一种先进的优化算法，适用于连续空间的非凸优化问题。
    通常需要较多的计算资源，但在一些困难问题上可能会表现得非常好。
    
    如果你的计算资源充足，网格搜索可能是一个可靠的选择，因为它可以穷举所有的参数组合来找到最优解。
    如果你希望在较短的时间内得到合理的解，随机搜索和贝叶斯优化可能是更好的选择。
    如果你面临的是一个非常复杂或非线性的问题，遗传算法和CMA-ES可能值得尝试。
    """
    
    optuna.logging.set_verbosity(optuna.logging.ERROR) 
#     def lgb_median_absolute_error(y_true, y_pred):
#         return 'median_absolute_error', median_absolute_error(y_true, y_pred), False 
    
#     lb = LabelBinarizer()
#     Y_lb = lb.fit_transform(Y) 
#     def custom_eval_metric(y_true, y_pred):
#         predicted_categories = np.argmax(y_pred, axis=1).reshape(-1, 1)
#         y_pred_cate = lb.inverse_transform(predicted_categories) 
        
#         y_pred_ = y_pred_cate.copy().astype(np.float64) 
#         y_true_ = y_true.copy().astype(np.float64)
#         return 'median_absolute_error', median_absolute_error(y_true_, y_pred_), False 

#     def custom_eval_metric2(y_true, y_pred):
#         y_pred_ = y_pred.copy().astype(np.float64) 
#         y_true_ = y_true.copy().astype(np.float64)
#         return  mean_absolute_error(y_true_, y_pred_)
    
    
    # tpe params
#     tpe_params = TPESampler.hyperopt_parameters()
#     tpe_params['n_startup_trials'] = 100 
    samplers = {
#                 'grid': optuna.samplers.GridSampler(), 
                'random': optuna.samplers.RandomSampler(), 
#                 'anneal': optuna.samplers, 
#                 'tpe': optuna.samplers.TPESampler(**tpe_params), 
                'tpe': optuna.samplers.TPESampler(), 
                'cma': optuna.samplers.CmaEsSampler(), 
                'nsgaii': optuna.samplers.NSGAIISampler()} 
#     optuna.logging.set_verbosity(optuna.logging.ERROR) 
    
    if isinstance(Y, pd.DataFrame):
        Y = Y.values.reshape(-1) 
    if objective_type == 'regression':
        objective_list = ['regression', 'regression_l1', 'quantile']  # ['regression', 'regression_l1', 'quantile','huber', 'mape']
    elif objective_type == 'binary':
        objective_list = ['binary'] 
    elif objective_type == 'multiclass':
        objective_list = ['softmax', 'multiclassova'] 
        
    def objective(trial): 
        #params
        param_grid = { 
                        'boosting_type': trial.suggest_categorical("boosting_type", ['gbdt', 'dart', 'rf']), # 'gbdt', 'dart', 'rf'
                        'objective': trial.suggest_categorical("objective", objective_list), 
                        "n_estimators": trial.suggest_int("n_estimators", 100, 5000), 
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
                        "num_leaves": trial.suggest_int("num_leaves", 2, 63),
                        "max_depth": trial.suggest_int("max_depth", 2, 31), 
                        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
                        'min_sum_hessian_in_leaf': trial.suggest_float("min_sum_hessian_in_leaf", 1e-8, 5, log=True), 
                        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
                        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
                        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 10, log=True),
                        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1, step=0.05),
                        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7), 
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1, step=0.02),
                        'feature_fraction_bynode':  trial.suggest_float("feature_fraction_bynode", 0.05, 1, step=0.02),
                        'max_bin': trial.suggest_int("max_bin", 63, 511), # 默认255 
                        'min_data_in_bin': trial.suggest_int("min_data_in_bin", 1, 20),
                        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 50), 
                        'seed': trial.suggest_int('seed', 0, 0), 
                        'n_jobs': trial.suggest_int('n_jobs', n_jobs_est, n_jobs_est) 
                     } 
        
        # CATE COLS 
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist() 
        if len(cat_cols)>0: 
            param_grid.update({"categorical_features": trial.suggest_categorical("categorical_features", [cat_cols]),
                              'cat_smooth': trial.suggest_float("cat_smooth", 1e-5, 200),
                               'cat_l2': trial.suggest_float("cat_l2", 1e-5, 200) 
                              }) 
        # other prms 
        if objective_type == 'regression':
            param_grid.update({"alpha": trial.suggest_float("alpha", 0.5, 0.5), 
                              'metric': trial.suggest_categorical("metric", ['', 'mae', 'mse', 
                                                                             'quantile', 'huber']), 
                              }) 

        elif objective_type == 'binary': 
            param_grid.update({'class_weight': trial.suggest_categorical("class_weight", ['balanced', None]), 
                              'metric': trial.suggest_categorical("metric", [
#                                   '', 'binary_logloss', 'average_precision', 
                                                                             'auc']) 
                              }) 
        elif objective_type == 'multiclass':
            param_grid.update({'class_weight': trial.suggest_categorical("class_weight", ['balanced', None]), 
                              'num_class': trial.suggest_int('num_class', np.unique(Y).shape[0], 
                                                            np.unique(Y).shape[0]),
                              'metric': trial.suggest_categorical("metric", [ 'multi_logloss'])  #  '',  'auc_mu', 'multi_error'
                              }) 
        else: 
            raise ValueError('objective_type error')
            
        if use_gpu == True:
            gpu_params = {'device_type': trial.suggest_categorical("device_type", ['gpu']),   # cpu, gpu, cuda 
                          'gpu_platform_id': trial.suggest_categorical("gpu_platform_id", [0]),  
                          'gpu_device_id': trial.suggest_categorical("gpu_device_id", [0])} 
            param_grid.update(gpu_params)
        
        # est : oof score平均得到
#         est_lgb = lgb.LGBMClassifier(**param_grid)
#         score = cross_val_score(est_lgb, X, Y, 
#                                 cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0), 
#                                 scoring=scoring).mean() 
#         return score 

    
        # 交叉验证
        if objective_type in ('binary', 'multiclass'):
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=100) 
            kf_split = kf.split(X, Y)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=0) 
            kf_split = kf.split(X)
        
        if objective_type in ('binary', 'multiclass'):
            if scoring in ('accuracy_score', 'balanced_accuracy_score'): 
                oof_pred = pd.Series([None]*X.shape[0]) 
            elif scoring in ('average_precision', 'roc_auc', 'log_loss', 'multi_logloss'):
                oof_pred = np.zeros([Y.shape[0], np.unique(Y).shape[0]]) 
            else:
                raise ValueError('check scoring')
        else:
            oof_pred = np.zeros(X.shape[0]) 

        for idx, (train_idx, test_idx) in enumerate(kf_split): 
            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx] 
            y_train, y_val = Y[train_idx], Y[test_idx] 
            if objective_type in ('binary', 'multiclass'):
                # LGBM建模
                model = lgb.LGBMClassifier(**param_grid, verbose=-2) 
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    #                 callbacks=[LightGBMPruningCallback(trial, "average_precision")]
                        ) 
            else:
                model = lgb.LGBMRegressor(**param_grid, verbose=-2) 
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    #                 callbacks=[LightGBMPruningCallback(trial, "average_precision")]
                        )
            # 预测
            if scoring in ('average_precision', 'roc_auc', 'log_loss', 'multi_logloss'):
                y_pred_prob = model.predict_proba(X_val, num_iteration=model.best_iteration_) 
                oof_pred[test_idx] = y_pred_prob 
            elif scoring in ('accuracy_score', 'balanced_accuracy_score'): 
                preds = model.predict(X_val, num_iteration=model.best_iteration_)
                oof_pred.iloc[test_idx] = preds 
            elif scoring in ('mean_absolute_error', 'mean_squared_error', 'median_absolute_error'): 
                preds = model.predict(X_val, num_iteration=model.best_iteration_)
                oof_pred[test_idx] = preds 
            else:
                raise ValueError('check scoring')
        
        if scoring == 'average_precision':
            lb = LabelBinarizer()
            # 字母序encoding
            Y_lb = lb.fit_transform(Y) 
            score = average_precision_score(Y_lb, oof_pred) 
        elif scoring == 'roc_auc':
#             lb = LabelBinarizer() 
#             Y_lb = lb.fit_transform(Y) 
#             print(oof_pred)  
            score = roc_auc_score(Y, oof_pred[:, 1]) 
        elif scoring in ('log_loss', 'multi_logloss'):
            lb = LabelBinarizer() 
            Y_lb = lb.fit_transform(Y) 
            score = log_loss(Y_lb, oof_pred) 
        elif scoring in ('accuracy_score'):
            score = accuracy_score(Y, oof_pred) 
        elif scoring in ( 'balanced_accuracy_score'):
            score = custom_eval_metric2(Y, oof_pred)
        elif scoring == 'mean_absolute_error':
            score = mean_absolute_error(Y, oof_pred)
        elif scoring == 'mean_squared_error':
            score = mean_squared_error(Y, oof_pred)
        elif scoring == 'median_absolute_error':
            score = median_absolute_error(Y, oof_pred) 
            score2 = mean_absolute_error(Y, oof_pred)
        else:
            raise ValueError('check scoring')  
        return score 


    study = optuna.create_study(direction=direction, 
                                sampler=samplers[sampler], 
#                                 pruner=optuna.pruners.HyperbandPruner(max_resource='auto'), 
#                                 pruner=optuna.pruners.MedianPruner(interval_steps=20, n_min_trials=8),
                                storage=optuna.storages.RDBStorage(url="sqlite:///db.sqlite3", 
                                                                   engine_kwargs={"connect_args": {"timeout": 500}}),  # 指定database URL 
                                study_name= '%s_%s_%s'%(study_name, sampler, X.shape[1]), load_if_exists=True) 
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True, n_jobs=n_jobs_optuna) 
    print('Best lgbm params:', study.best_trial.params) 
    print('Best CV score:', study.best_value) 
    
    optuna.visualization.plot_optimization_history(study).show() 
#     optuna.visualization.plot_intermediate_values(study).show() 
#     optuna.visualization.plot_parallel_coordinate(study).show() 
#     optuna.visualization.plot_parallel_coordinate(study, params=["max_depth", "min_samples_leaf"]).show() 
#     optuna.visualization.plot_contour(study).show() 
#     optuna.visualization.plot_contour(study, params=["max_depth", "min_samples_leaf"]).show() 
#     optuna.visualization.plot_slice(study).show() 
    optuna.visualization.plot_param_importances(study).show() 
    params_df = study.trials_dataframe() 
    params_df = params_df.sort_values(by=['value']) 
    return params_df, study 


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




