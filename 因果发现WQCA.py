#!/usr/bin/env python
# coding: utf-8

# In[171]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import r2_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2023) 
from econml.dml import CausalForestDML 


# ##### 列分组

# In[2]:


train = pd.read_csv('./dataset/data/train.csv')
test = pd.read_csv('./dataset/data/test.csv') 


# ##### 寻找A

# In[3]:


col_std = train.std(axis=0)
col_std[col_std==0]


# In[4]:


# 这2列不包含任何信息，属于A
A = ['V_20', 'V_27']


# In[5]:


# 相关系数
tr_corr = train.corr() 
# 去掉对角列
tr_corr[tr_corr==1] = np.nan
# 找到每列与其余列最大的相关系数
tr_corr_max = tr_corr.max(axis=0) 
# 相关系数最大不超过0.02，可以认为属于独立变量
tr_corr_max[tr_corr_max<0.02]


# In[6]:


A.extend(['V_17', 'V_38'])


# In[7]:


A


# In[56]:


tr_corr.abs().mean()


# ##### 填充空值

# In[8]:


train = train.drop(columns=A) 
test = test.drop(columns=A) 


# In[9]:


# 类别变量
cateCols = ['V_8','V_10','V_14','V_26']


# In[10]:


for c in cateCols:
    print(train[c].unique(), test[c].unique()) 


# In[11]:


# 可以将unknown暂时设为空值Null, 之后进行预测与填充
for c in cateCols:
    train[c] = train[c].replace({'unknown': None})
    test[c] = test[c].replace({'unknown': None})


# In[12]:


class PredNan():
    def __init__(self):
        """
        如果test也有空值，train test concat后进行空值填充
        y_type: regression, binary, multiclass  
        确保数据类型:  int, float, category
        """
        self.y_type = None
    def GridSearchCVParams(self):
        # 定义模型
        if self.y_type in ('multiclass', 'binary'): 
            params = {'boosting_type': 'gbdt', 'objective': self.y_type,
                    'class_weight': 'balanced', 'n_jobs': -1} 
            estimator = lgb.LGBMClassifier(**params) 
            param_grid = {'learning_rate': [0.03], 'n_estimators': [700, 1000, 2000],
                            'num_leaves': [7, 15, 31, 50],  'min_child_samples':[1, 3, 5, 7 , 11], 'seed': [42]} 
            scoring = 'roc_auc_ovr_weighted'
        else:
            params = {'boosting_type': 'gbdt', 'objective': 'regression', 'n_jobs': -1}
            estimator = lgb.LGBMRegressor(**params) 
            param_grid = {'learning_rate': [0.03],  'n_estimators': [700, 1200, 2000],
                'num_leaves': [7, 15, 31, 50],  'min_child_samples':[1, 2, 4, 7, 10, 15],  'seed': [42]} 
            scoring = 'r2'
        # 使用GridSearchCV进行交叉验证
        grid = GridSearchCV(estimator, param_grid, cv=3, scoring = scoring, n_jobs=-1)
        grid.fit(self.X, self.Y) 
        # 输出最佳参数
        print('Best parameters found by grid search are:', grid.best_params_)
        params.update(grid.best_params_)
        return params

    def fillna(self, data):
        self.data = data
        self.cols = self.data.columns
        for c in self.cols:
            if self.data[c].count() != self.data.shape[0]:
                if self.data[c].dtype == "float" or self.data[c].dtype == "int":
                    self.y_type = 'regression'
                else:
                    if self.data[c].unique().shape[0] -1 ==2:
                        self.y_type = 'binary'
                    elif self.data[c].unique().shape[0] -1 > 2: 
                        self.y_type = 'multiclass'
                    else:
                        raise ValueError('类别变量category=1')
            
                self.Y = self.data.loc[self.data[c].notnull()][c]
                self.X = self.data.loc[self.data[c].notnull()].drop(columns=[c])
                self.X_toPred = self.data.loc[self.data[c].isnull()].drop(columns=[c])
                print(c, self.y_type) 
                params = self.GridSearchCVParams()
                estimator = lgb.LGBMClassifier(**params) if self.y_type in ('multiclass', 'binary') else lgb.LGBMRegressor(**params) 
                estimator.fit(self.X, self.Y)
                r = estimator.predict(self.X_toPred)
                print(c, self.data.loc[self.data[c].isnull()][c].unique(), self.X_toPred.shape, r.shape)
                self.data.loc[self.data[c].isnull(), c] = r 
                print(c, self.data.loc[self.data[c].isnull()][c].shape, self.data.loc[self.data[c].isnull()][c].unique())
#         return self.data 


# In[13]:


# 测试集中也有空值，一起进行填充


# In[14]:


data = pd.concat((train.iloc[:, :-2], test)) 


# In[15]:


# 类别变量设置为category
for c in cateCols:
    data[c] = data[c].astype('category')


# In[16]:


filler = PredNan()
filler.fillna(data)


# In[25]:


data.to_csv('./dataset/data/filledData.csv', index=False) 


# In[23]:


train, test = data[:-5000], data[-5000:]


# In[26]:


T_Y = pd.read_csv('./dataset/data/train.csv', usecols=['treatment', 'outcome']) 


# In[82]:


train['V_10'].values.unique().shape[0]


# In[106]:


def scoring(X, Y):
    if Y.dtype != 'category':
        objective = 'regression'
        params = {'boosting_type': 'gbdt', 'objective': objective, 'n_jobs': -1}
        estimator = lgb.LGBMRegressor(**params) 
        scoring = 'neg_root_mean_squared_error'
    else:
        objective = 'binary' if Y.unique().shape[0] ==2 else 'multiclass'
        params = {'boosting_type': 'gbdt', 'objective': objective, 'n_jobs': -1}
        estimator = lgb.LGBMClassifier(**params) 
        scoring = 'roc_auc_ovr_weighted'
        le_y = LabelEncoder()
        Y = le_y.fit_transform(Y)
    param_grid = {'learning_rate': [0.03],
                  'n_estimators': [1000, 2000],
                  'num_leaves': [7, 15, 25, 31],
                  'min_child_samples':[1, 3, 5, 10, 15],
                  'seed': [42]} 
    grid = GridSearchCV(estimator, param_grid, cv=3, scoring=scoring, n_jobs=-1)
    grid.fit(X, Y) 
    params.update(grid.best_params_) 
    estimator = lgb.LGBMRegressor(**params) if objective == 'regression' else lgb.LGBMClassifier(**params) 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2023)
    estimator.fit(X_train, y_train) 
    x_pred = estimator.predict(X_test)
    if objective == 'regression':
        r2, mse = round(r2_score(y_test, x_pred), 4) , round(mean_squared_error(y_test, x_pred), 4) 
        return r2, mse
    else:
        x_proba = estimator.predict_proba(X_test)
        accu, auc = round(balanced_accuracy_score(y_test, x_pred), 
                          4), round(roc_auc_score(y_test, x_proba[:, 1], average='weighted'), 4)
        return accu, auc


# In[ ]:





# ##### T和Y作为特征预测某一列a，如果再加上某一列b作为特征，预测结果变好，说明a与b在T和Y的同侧

# In[32]:


ohe = OneHotEncoder(drop='first', sparse_output=False)
T_ohe = ohe.fit_transform(T_Y[['treatment']])
T_Y = np.concatenate((T_ohe, T_Y[['outcome']].values), axis=1)
# poly features 
featurizer = PolynomialFeatures(degree=3)
T_Y_feature = featurizer.fit_transform(T_Y) 


# In[33]:


T_Y_feature.shape


# In[35]:


contCols = [i for i in train.columns if i not in cateCols] 


# ##### 选一列比如V_7(与其他列的相关性平均值较大的)作为被预测列

# In[37]:


# target
target = train['V_7'].values 
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 


# In[39]:


base_r2, base_mse


# In[41]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[42]:


# CATEGORY 
for col in cateCols:
    col_ = train[[col]].values
    oht_ = OneHotEncoder(drop='first', sparse_output=False)
    col_ = oht_.fit_transform(col_)
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[ ]:


# 暂定将R2差值>0.03认为与V_7同组, R2差值<0.002的认为与V_7不同组, 中间的认为不能确定
# 可以看到feature之间互相的相关性联系比较多，因此几次尝试就能将fearture区分开
GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_15', 'V_16', 'V_19', 'V_24', 'V_25', 'V_29', 'V_31', 'V_36', 'V_37', 'V_39', 'V_8', 'V_14']
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_23', 'V_30', 'V_32', 'V_33', 'V_34', 'V_26', 'V_10']
TOBD = ['V_12', 'V_21', 'V_22', 'V_28', 'V_35'] 


# ##### 再选取一个col  V_15(与其他列的相关性平均值较大的) 作为target, 重复上面过程

# In[60]:


# target
target = train['V_15'].values
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 
base_r2, base_mse 


# In[61]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[62]:


# CATEGORY 
for col in cateCols:
    col_ = train[[col]].values
    oht_ = OneHotEncoder(drop='first', sparse_output=False)
    col_ = oht_.fit_transform(col_)
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[ ]:


# 可以确定V_12, V_21, V_35 属于GROUP_NOT7, 
# 可以确定V_28, V_10 属于GROUP7
GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_21', 'V_24', 'V_25', 'V_28', 'V_29', 'V_31', 'V_35','V_36', 'V_37', 'V_39', 'V_8', 'V_14']
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_23', 'V_30', 'V_32', 'V_33', 'V_34', 'V_26', 'V_10']
TOBD = ['V_22'] 


# In[65]:


# 只剩下V_22, 将V_22作为target


# In[63]:


# target
target = train['V_22'].values
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 
base_r2, base_mse 


# In[64]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[66]:


# 可以看到 V_22与V_4, V_18, V_23, V_29, V_34有较强相关性, 因此V_22属于GROUP_NOT7
# V_29与V_7有R2=0.04相关性, 且与V_22有0.08相关性，需要再次测试来确定属于哪个组
GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_21', 'V_24', 'V_25', 'V_28', 'V_29', 'V_31', 'V_35','V_36', 'V_37', 'V_39', 'V_8', 'V_14']
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_22', 'V_23', 'V_30', 'V_32', 'V_33', 'V_34', 'V_26', 'V_10']
TOBD = ['V_29']


# In[68]:


# 以V_29为target
# target
target = train['V_29'].values
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 
base_r2, base_mse 


# In[69]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[71]:


# CATEGORY 
for col in cateCols:
    col_ = train[[col]].values
    oht_ = OneHotEncoder(drop='first', sparse_output=False)
    col_ = oht_.fit_transform(col_)
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[70]:


# V_29与 V_18, V_33相关性最高，因此V_29属于GROUP_NOT7
GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_21', 'V_24', 'V_25', 'V_28', 'V_29', 'V_31', 'V_35','V_36', 'V_37', 'V_39', 'V_8', 'V_14']
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_22', 'V_23', 'V_29','V_30', 'V_32', 'V_33', 'V_34', 'V_26', 'V_10']


# In[ ]:





# ##### 多次随机选择测试，以便得到更准确的结果

# In[107]:


# 以V_10为target
target = train['V_10'].values
base_accuracy, base_auc = scoring(X=T_Y_feature, Y=target) 
base_accuracy, base_auc


# In[111]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    accu, auc = scoring(X, target) 
    print(col, '正确率', base_accuracy, '添加后', accu, '差值:', accu - base_accuracy , '  |  ', 'AUC', base_auc, '添加后', auc,  '差值：', auc-base_auc)


# In[124]:


# 以V_10为target
target = train['V_10'].values
base_accuracy, base_auc = scoring(X=T_Y_feature, Y=target) 
base_accuracy, base_auc


# In[125]:


# CATEGORY 
for col in cateCols:
    col_ = train[[col]].values
    oht_ = OneHotEncoder(drop='first', sparse_output=False)
    col_ = oht_.fit_transform(col_)
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    accu, auc = scoring(X, target) 
    print(col, '正确率', base_accuracy, '添加后', accu, '差值:', accu - base_accuracy , '  |  ', 'AUC', base_auc, '添加后', auc,  '差值：', auc-base_auc)


# In[128]:


# 正确率binary=0.5004, 可以判断V_10与T,Y基本无相关性，V_10与其他变量的相关性都非常小,因此V_10属于A的可能性很大。
#但是比较其中相关性最大的几个是V_7, V_12, V_23, V_25, V_28, V_31,这几个基本都属于GROUP7，因此也有可能属于GROUP7，暂时归为GROUP7
GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_21', 'V_24', 'V_25', 'V_28', 'V_29', 'V_31', 'V_35','V_36', 'V_37', 'V_39', 'V_8', 'V_14', 'V_10']  
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_22', 'V_23', 'V_29','V_30', 'V_32', 'V_33', 'V_34', 'V_26']


# In[ ]:





# In[ ]:





# ##### 对于A中['V_17', 'V_38'], 之前是基于线性相关性的判断, 可能是不准确的。有必要再使用模型进行分析

# In[114]:


V17V38 = pd.read_csv('./dataset/data/train.csv', usecols=['V_17', 'V_38'])


# In[117]:


# 以V_17为target
# target
target = V17V38['V_17'].values
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 
base_r2, base_mse 


# In[118]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[119]:


# R2=-0.0039,说明V_17与T,Y无关, 而且可以看到差值几乎都是负值，即每个特征相对于V_17来说都是噪音，只增加了过拟合。因此可以判断V_17属于A.


# In[120]:


# 以V_38为target
# target
target = V17V38['V_38'].values
base_r2, base_mse = scoring(X=T_Y_feature, Y=target) 
base_r2, base_mse 


# In[121]:


# CONTINUOUS 
for col in contCols:
    col_ = train[[col]].values
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[122]:


# CATEGORY 
for col in cateCols:
    col_ = train[[col]].values
    oht_ = OneHotEncoder(drop='first', sparse_output=False)
    col_ = oht_.fit_transform(col_)
    poly = PolynomialFeatures(degree=3) 
    X = poly.fit_transform(np.concatenate((T_Y, col_), axis=1)) 
    r2, mse = scoring(X, target) 
    print(col, 'R2', base_r2, '添加后', r2, '差值:', r2-base_r2 , '  |  ', 'MSE', base_mse, '添加后', mse,  '差值：', mse-base_mse)


# In[123]:


# R2=-0.0085,说明V_38与T,Y无关, 而且可以看到差值都在0.004以下, 与任何一个特征都没有明显的相关关系, 因此可以判断V_17属于A.


# In[ ]:





# ##### 所有变量已分组。对于 GROUP7与GROUP_NOT7，只需要分别作为W进行一次因果effect测试，查看score即可判断GROUP7是WQ，GROUP_NOT7是C。

# In[ ]:


GROUP7 = ['V_7', 'V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_21', 'V_24', 'V_25', 'V_28', 'V_29', 'V_31', 'V_35','V_36', 'V_37', 'V_39', 'V_8', 'V_14', 'V_10']  
GROUP_NOT7 = ['V_0', 'V_1', 'V_3', 'V_4', 'V_11', 'V_13', 'V_18', 'V_22', 'V_23', 'V_30', 'V_32', 'V_33', 'V_34', 'V_26']


# In[129]:


WQ = GROUP7
C = GROUP_NOT7


# In[ ]:





# ##### 分离Q

# In[127]:


# Q混在W中，会造成过拟合等问题，因此有必要进行分离


# In[134]:


X = pd.get_dummies(train[WQ], sparse=False, drop_first=True) 


# In[135]:


X_test = pd.get_dummies(test[WQ], sparse=False, drop_first=True) 


# In[137]:


T = pd.read_csv('./dataset/data/train.csv', usecols=['treatment']) 
T['treatment'] = T['treatment'].astype(str).rename(columns={'treatment': 'T'}) 


# In[140]:


Y = pd.read_csv('./dataset/data/train.csv', usecols=['outcome']).rename(columns={'outcome': 'Y'})


# In[150]:


Y.shape, X.shape, T.shape, X_test.shape


# In[154]:


XTY = pd.concat((X, T, Y), axis=1)

XTY01 = XTY.loc[XTY['T'].isin(['0', '1'])] 
XTY02 = XTY.loc[XTY['T'].isin(['0', '2'])]  

XTY01['T'] = XTY01['T'].astype('category')
XTY02['T'] = XTY02['T'].astype('category') 

X_01 = XTY01.drop(columns=['T', 'Y'])
T_01 = XTY01[['T']]
Y_01 = XTY01[['Y']]
X_02 = XTY02.drop(columns=['T', 'Y'])
T_02 = XTY02[['T']]
Y_02 = XTY02[['Y']] 


# In[193]:


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
        param_grid = {'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(500, 3000),
                      'num_leaves': Integer(5, 63),
                      'min_child_samples': Integer(1, 40), 
#                       'reg_alpha': Real(0, 0.1),
#                       'reg_lambda': Real(0, 0.2), 
                      'seed': Categorical([42])} 
    else:
        params = {'boosting_type': 'gbdt',  'n_jobs': -1}
        estimator = lgb.LGBMRegressor(**params) 
        param_grid = {'objective': Categorical(['regression', 'regression_l1', 'huber', 'fair']),
                      'learning_rate': Real(0.01, 0.05), 
                      'n_estimators': Integer(500, 3000),
                      'num_leaves': Integer(4, 50),  
                      'min_child_samples': Integer(1, 50),
#                       'reg_alpha': Real(0, 0.2),
#                       'reg_lambda': Real(0, 0.2), 
                      'seed': Categorical([42])} 
        
    # search 
    print(scoring) 
    grid = BayesSearchCV(estimator, param_grid, 
                         n_iter=200,
                         cv=3, scoring = scoring, n_jobs=-1, verbose=0)
    grid.fit(X, Y) 
    params.update(grid.best_params_) 
    print('Best parameters found by grid search are:', params)
    print('best Score', grid.best_score_)
    return params 


# In[188]:


params_XY01 = bayesGridSearchCVParams(X_01, Y_01, objective='regression', scoring='neg_root_mean_squared_error') 
mdl_y01 = lgb.LGBMRegressor(**params_XY01) 
# {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05, 'min_child_samples': 20, 'n_estimators': 3000, 'num_leaves': 10, 'objective': 'regression', 'seed': 42}
# best Score -3.49207942347328


# In[192]:


params_XT01 = bayesGridSearchCVParams(X_01, T_01, objective='binary', scoring='roc_auc_ovr_weighted') 
mdl_t01 = lgb.LGBMClassifier(**params_XT01) 
# {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.01, 'min_child_samples': 5, 'n_estimators': 500, 'num_leaves': 7, 'seed': 42}
# best Score 0.8295223110391433


# In[194]:


params_XY02 = bayesGridSearchCVParams(X_02, Y_02, objective='regression', scoring='neg_root_mean_squared_error') 
mdl_y02 = lgb.LGBMRegressor(**params_XY02) 
# {'boosting_type': 'gbdt', 'n_jobs': -1, 'learning_rate': 0.05, 'min_child_samples': 25, 'n_estimators': 3000, 'num_leaves': 10, 'objective': 'regression', 'seed': 42}
# best Score -6.684132697555242


# In[195]:


params_XT02 = bayesGridSearchCVParams(X_02, T_02, objective='binary', scoring='roc_auc_ovr_weighted') 
mdl_t02 = lgb.LGBMClassifier(**params_XT02) 
# {'boosting_type': 'gbdt', 'objective': 'binary', 'class_weight': 'balanced', 'n_jobs': -1, 'learning_rate': 0.01, 'min_child_samples': 20, 'n_estimators': 1000, 'num_leaves': 5, 'seed': 42}
# best Score 0.8081530413332749


# In[214]:


def calc_score_all(train_t1, train_t2, test_data_t1, test_data_t2):
    tr = np.concatenate((train_t1.reshape(-1, 1), train_t2.reshape(-1, 1)), axis=1)
    tr = pd.DataFrame(tr, columns=['causalTree1', 'causalTree2'])
    target = pd.read_csv('./dataset/data/target.csv')
    cau_tree = pd.DataFrame({'causalTree1': test_data_t1, 
                            'causalTree2': test_data_t2}) 
    cau_tree = pd.concat([tr, cau_tree], axis=0)
    cau_tree = cau_tree.reset_index(drop=True) 
    target = target.reset_index(drop=True) 
    result = pd.concat([target, cau_tree], axis=1)
    def cal_metric(result):
        r = np.sqrt(np.sum((result.ce_1 - result.causalTree1)**2)/result.shape[0])/result.ce_1.mean() +             np.sqrt(np.sum((result.ce_2 - result.causalTree2)**2)/result.shape[0])/result.ce_2.mean()
        return r 
    return cal_metric(result) 


def effectScoring(drop_col=[]):
    n_est = 500
    min_samp_split = 200
    max_feat = 0.9
    max_dep = 6
    max_samp = 0.3
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
                    min_samples_split = min_samp_split, 
                    max_samples = max_samp, 
                    honest=True, 
    #                                     min_weight_fraction_leaf = 0.01 ,
    #                                     min_var_leaf_on_val = False , 
                    inference =True,
                    max_features = max_feat,
                    n_jobs =-1, 
                    random_state =2023,
                    verbose =0 ) 

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
                    min_samples_split = min_samp_split, 
                    max_samples = max_samp,
                    honest=True, 
    #                                     min_weight_fraction_leaf = 0.01 ,
    #                                     min_var_leaf_on_val = False , 
                    inference =True,
                    max_features = max_feat,
                    n_jobs =-1, 
                    random_state =2023,
                    verbose =0 ) 
    est01.fit(Y=Y_01, T=T_01, X=X_01.drop(columns=drop_col))
    est02.fit(Y=Y_02, T=T_02, X=X_02.drop(columns=drop_col))
    test_t01 = est01.effect(X=X_test.drop(columns=drop_col), T0='0', T1='1') 
    test_t02 = est02.effect(X=X_test.drop(columns=drop_col), T0='0', T1='2') 
    X_t01 = est01.effect(X=X.drop(columns=drop_col), T0='0', T1='1')  
    X_t02 = est02.effect(X=X.drop(columns=drop_col), T0='0', T1='2') 
    score = calc_score_all(X_t01.reshape(-1), X_t02.reshape(-1), test_t01.reshape(-1), test_t02.reshape(-1)) 
    impt = pd.DataFrame({'feature': X.drop(columns=drop_col).columns, 'est01_impt': est01.feature_importances().reshape(-1), 'est02_impt': est02.feature_importances().reshape(-1)})
    return score, impt 


# ##### 不删除特征，作为baseline

# In[215]:


score, impt = effectScoring(drop_col=[]) 
score


# In[221]:


impt['mean_'] = impt.mean(axis=1)


# In[224]:


impt.sort_values(by=['mean_'], ascending=False) 


# In[226]:


# 寻找去掉后会让score(rmse)上升的col


# In[228]:


for COL in X.columns:
    score_, _ = effectScoring(drop_col=[COL])
    print(COL, score, score_, score - score_) 


# In[230]:


# 选出去掉后score上升的col,是W
W = ['V_2', 'V_5', 'V_6', 'V_9', 'V_12', 'V_15', 'V_16', 'V_19', 'V_28', 'V_39', 'V_8_yes', 'V_14_yes'] 


# In[232]:


Q = list(set(X.columns) - set(W)) 
Q


# In[231]:


# 测试W中是否还有可以去掉的col


# In[235]:


# 特征变化较大，重新寻找DML模型参数
params_XY01 = bayesGridSearchCVParams(X_01.drop(columns=Q), Y_01, objective='regression', scoring='neg_root_mean_squared_error') 
mdl_y01 = lgb.LGBMRegressor(**params_XY01) 
params_XT01 = bayesGridSearchCVParams(X_01.drop(columns=Q), T_01, objective='binary', scoring='roc_auc_ovr_weighted') 
mdl_t01 = lgb.LGBMClassifier(**params_XT01) 
params_XY02 = bayesGridSearchCVParams(X_02.drop(columns=Q), Y_02, objective='regression', scoring='neg_root_mean_squared_error') 
mdl_y02 = lgb.LGBMRegressor(**params_XY02) 
params_XT02 = bayesGridSearchCVParams(X_02.drop(columns=Q), T_02, objective='binary', scoring='roc_auc_ovr_weighted') 
mdl_t02 = lgb.LGBMClassifier(**params_XT02) 


# In[236]:


# baseline
score, _ = effectScoring(drop_col= Q) 
score 


# In[238]:


for COL in W:
    score_, _ = effectScoring(drop_col= Q+[COL])
    print(COL, score, score_, score - score_) 


# In[239]:


# V_12会使得效果变差，V_12放入Q
Q.append('V_12')


# In[272]:


Q


# In[241]:


# 继续尝试一个一个加入Q中的列，看是否有可以提升的col


# In[303]:


# baseline
score, _ = effectScoring(drop_col= Q) 
score 


# In[243]:


for COL in Q:
    score_, _ = effectScoring(drop_col= list(set(Q)-set([COL])))
    print(COL, score, score_, score - score_) 


# In[244]:


# 加入Q中特征没有提升


# In[245]:


# 尝试加入C中特征


# In[316]:


C_train = train[C]
C_test = test[C] 
C_train = pd.get_dummies(C_train, sparse=False, drop_first=True) 
C_test = pd.get_dummies(C_test, sparse=False, drop_first=True) 
XTY_C = pd.concat([C_train, T, Y], axis=1) 

XTY_C01 = XTY_C.loc[XTY_C['T'].isin(['0', '1'])] 
XTY_C02 = XTY_C.loc[XTY_C['T'].isin(['0', '2'])]  

XTY_C01['T'] = XTY_C01['T'].astype('category')
XTY_C02['T'] = XTY_C02['T'].astype('category') 

X_01_C = XTY_C01.drop(columns=['T', 'Y'])
T_01_C = XTY_C01[['T']]
Y_01_C = XTY_C01[['Y']]
X_02_C = XTY_C02.drop(columns=['T', 'Y'])
T_02_C = XTY_C02[['T']]
Y_02_C = XTY_C02[['Y']] 
C.remove('V_26')
C.append('V_26_yes') 


# In[321]:


C_train.shape, C_test.shape, X_01_C.shape, T_01_C.shape, Y_01_C.shape, X_02_C.shape, T_02_C.shape, Y_02_C.shape


# In[332]:


for COL in C:
    X_01 = pd.concat([X_01, X_01_C[[COL]]], axis=1)
    X_02 = pd.concat([X_02, X_02_C[[COL]]], axis=1)
    X_test = pd.concat([X_test, C_test[[COL]]], axis=1) 
    X = pd.concat([X, C_train[[COL]]], axis=1)
    score_, _ = effectScoring(drop_col= Q)
    print(COL, score, score_, score - score_) 
    X_01 = X_01.drop(columns=[COL])
    X_02 = X_02.drop(columns=[COL])
    X_test = X_test.drop(columns=[COL])
    X = X.drop(columns=[COL])
    


# In[338]:


# C中的V_22有充分的理由属于C, 且增加V_22对结果的提升很小, V_0,V_26_yes同理, 因此C中没有判断错误的列, 没有可以加入W的列。


# In[341]:


# 最终结果
W, Q, C


# ##### 标准化后保存最优特征

# In[345]:


std_scaler = StandardScaler() 
X_W_std = std_scaler.fit_transform(XTY[W]) 


# In[348]:


pd.DataFrame(X_W_std, columns=W).to_csv('./dataset/data/best2/X.csv') 
X_test_W_std = std_scaler.transform(X_test[W]) 
pd.DataFrame(X_test_W_std, columns=W).to_csv('./dataset/data/best2/test.csv') 


# In[356]:


# WQCA分类完成
# 分类结果与之前尝试的结果(./dataset/data/best)有些不同，少V_31, V_10_no, 多了V_6, V_16, 可能原因是这些特征间本身有共线性，可以互相代替，在DML参数不同的条件下，会选出不同的特征
# 从上面出来的结果看，再微调参数后，最终结果应该会不错


# In[ ]:




