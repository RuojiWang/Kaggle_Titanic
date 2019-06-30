#coding=utf-8
#这个版本是可以运行了，但是并没有产生特征我觉得真的很奇怪
#还有就是这个只是创造特征，所有创造的特征最后需要进行一次特征选择咯
#至于特征选择就是使用线性模型加上L1正则化进行选择咯
#https://stackoverflow.com/questions/52418152/featuretools-can-it-be-applied-on-a-single-table-to-generate-features-even-when
#下面的代码主要是根据上面的链接修改的，因为上面的代码不能够直接运行我也不知道是什么缘故
#下面的代码虽然可以运行了但是和我的预期不符，现在我似乎找到了原因到底在哪里，不应该用agg_primitives而应该用trans_primitives
#所以之前stackoverflow的代码不能够直接运行的缘故也就是tans_primitives的参数错了，可能这个库的版本变了吧参数也改了
from sklearn.datasets import load_iris
import pandas as pd
import featuretools as ft

# Load data and put into dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Make an entityset and add the entity
es = ft.EntitySet(id = 'iris')
es.entity_from_dataframe(entity_id = 'data', dataframe = df, 
                         make_index = True, index = 'index')
print(es["data"].df)

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data', max_depth=2, agg_primitives = ["max", "skew", "min", "mean", "count"])
print(df.shape)
print(feature_matrix.head())
print(feature_matrix.shape)
print(feature_defs)