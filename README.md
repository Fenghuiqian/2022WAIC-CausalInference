# 2022WAIC-CausalInference


## 一个因果推理竞赛
2022 WAIC 黑客松九章云极赛道-因果学习和决策优化挑战赛  

地址: https://tianchi.aliyun.com/competition/entrance/532019/rankingList

## 过程

### 因果发现
![因果图](https://img.alicdn.com/imgextra/i4/O1CN01VNDxpX1xOp4KeeFS6_!!6000000006434-2-tps-271-206.png)
1. 40个特征之间的相关关系及方向比较复杂，PC/LiNGAM等因果发现方法，难以观察到确定性的结论。
2. pearson/X2/entropy等方法也难以检验多特征之间复杂的相关关系。
3. 需要使用模型方法进行因果发现与特征区分，可以得到明确的结论。在用于数据生成的因果图中，W,Q与C被X,Y分开，因此如果使用X,Y为特征去拟合W,Q中的变量，添加C中的变量到特征中一定不会对拟合结果有提升，因为C中信息已经包含在了XY中，使用这个结论，可以把变量分成WQ组和C组。
4. WQ中Q的存在对因果推理没有作用，太多的Q混在特征中还会使得模型过拟合，需要去掉。使用模型方法，如果去掉某个变量对推理结果没有影响或者会提升效果，那么这个特征就属于Q。
5. 具体见代码: 因果发现WQCA.py。

### 因果推理

1. 不同的领域，因果推理的变量命名会有不同。在econml工具包中，X代表异质性特征(协变量)，W代表混淆变量(控制变量)，X和W对应于上图中的W。T代表治疗变量，对应于上图中的X。Y代表结果变量，对应于上图中的Y。上图中的Q,C是对因果推理无效应该去除的变量。A是独立变量，也要去除。
2. 异质性特征X(与CATE类别平均处理效应直接相关的变量，与Y直接相关，但不一定与T相关)和混淆变量W(是会直接影响T和Y的变量，但是不希望观察它们的CATE效应)即使在应用场景中有时也难以区分，使用DML模型，可以有效减少将混淆变量引入与CATE异质性无关的变量中引起的误差。区分XW的代码测试了X中哪些属于W，正确分类XW对样本的effect结果有微小的提升。
3. 使用CausalForestDML，ForestDRLearner，DRLearner三种模型，score分别为0.36, 0.41, 0.55.
4. stacking后的结果是0.20，好于2022年的冠军方案0.28(多个DeepLearning模型的stacking).
5. T01和T02分开预测，对结果有较大提升(应该是与赛题数据生成过程及模型对类别Treatment的处理方式有关)。
6. 空值预测填充，对结果有提升。
7. IsoForest去除1%离群点，对结果有很小提升。
8. 是数值变量但数值种类很少的连续特征，当作类别变量处理，对结果没影响。
9. T01不平衡样本，欠采样/过采样，结果都会变差，注意类别权重即可。


## 环境

python = 3.9.17

econml = 0.14.1  

lightgbm = 4.0.0  

sklearn = 1.2.2  


## 运行

1. Run 因果发现WQCA.py
2. Run CausalForestDML.py, ForestDRLearner.py, DRLearner.py
3. Run stack.py
