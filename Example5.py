#coding=utf-8
#我觉得这个遗传算法最大的问题就是漫无目的的搜索导致计算量过大速度过慢
#这个TPOT库确实是可以优化参数的，这个是遗传算法性质就可以决定的
#这个库作用主要在于（1）搜索模型。（2）搜索超参。（1）对我没用（2）hyperopt一样能做且效率更高
#这个库不仅不支持神经网络的模型，而且还不能够选择模型进行优化，这样效率太低了吧
#还好我运行了一下这个库，卧槽这个完全超出我的想象力呀，这个库是真的挺棒的，可以双修
#不过我还是觉得end_of_the_titanic4.py应该上限更高一点。毕竟神经网络模型的超参搜索
#我现在有点担心这个TPOT出来的结果有过拟合风险呢？end_of_the_titanic4是有early stopping的所以不担心
#end_of_the_titanic4有early stopping是否不再需要split数据集了呢？TPOT怕是必须split数据集的吧
#我个人觉得TPOT出来的结果可能存在过拟合的风险，随机森林拟合的效果才是夸张随笔都是90%以上的
#所以我觉得end_of_the_titanic4的也可以split一个版本吧，但是TPOT是必须split数据的吧
#TPOT有split数据和没有split数据确实不一样，就是说split出来的数据准确率不高，可能存在过拟合风险咯。
#https://github.com/topics/feature-engineering 下面的地址当中还只有一个机器学习库还没有搞定咯。。
import sys
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_validation

from sklearn.svm import SVC
from collections import Counter
from xgboost import XGBClassifier
from sklearn.utils import resample, shuffle
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection.rfe import RFECV
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.cross_validation import StratifiedKFold, cross_val_predict,cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel, RFE
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model.logistic import LogisticRegression

data_train = pd.read_csv("C:/Users/win7/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/win7/Desktop/test.csv")
combine = [data_train, data_test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)   

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['FamilySizePlus'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 2, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 3, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 4, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 5, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 6, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 7, 'FamilySizePlus'] = 1

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)
    
for dataset in combine: 
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0 
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1 
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2 
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3 
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
#这里的mode是求解pandas.core.series.Series众数的第一个值（可能有多个众数）
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#将data_test中的fare元素所缺失的部分由已经包含的数据的中位数决定哈
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine:
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = 0
    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = 1

#尼玛给你说的这个是贡献船票，原来的英文里面根本就没有这种说法嘛
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
#print(df)
df_ticket = df.index.values          #共享船票的票号
tickets = data_train.Ticket.values   #所有的船票
#print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #遍历所有船票，在共享船票里面的为1，否则为0
    result.append(ticket)
    
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          #共享船票的票号
tickets = data_train.Ticket.values   #所有的船票

result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #遍历所有船票，在共享船票里面的为1，否则为0
    result.append(ticket)

results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_train = pd.concat([data_train, results], axis=1)

df = data_test['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          
tickets = data_test.Ticket.values   
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_test = pd.concat([data_test, results], axis=1) 

data_train_1 = data_train.copy()
data_test_1  = data_test.copy()
data_test_1 = data_test_1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1)

X_train = data_train_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]
Y_train = data_train_1['Survived']

X_test = data_test_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]

X_all = pd.concat([X_train, X_test], axis=0)
#我觉得训练集和测试集需要在一起进行特征缩放，所以注释掉了原来的X_train的特征缩放咯
X_all_scaled = pd.DataFrame(preprocessing.scale(X_all), columns = X_train.columns)
#X_train_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)
X_train_scaled = X_all_scaled[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]

"""
#下面输出的文字怎么一会儿是白色的一会儿是红色的呢，这尼玛什么情况？
#这下面是有split数据的版本咯，果然有split数据就会原形毕露咯
#0.859973579921 0.783582089552
#0.846763540291 0.84328358209
#0.849405548217 0.820895522388
#0.842800528402 0.820895522388
#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
#                                                    train_size=0.75, test_size=0.25)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_scaled, Y_train, train_size=0.85, test_size=0.15)
skf = StratifiedKFold(y_train1, n_folds=5, shuffle=True, random_state=None)

#这个verbosity设置为2才能够看到每一代的输出，设置为1只有结果咯
tpot = TPOTClassifier(generations=20, population_size=100, verbosity=2,
                      cv=skf, config_dict='TPOT light')
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_mnist_pipeline.py')
tpot.fit(X_train1, y_train1)
print(tpot.score(X_train1, y_train1))
print(tpot.score(X_test1, y_test1))
#这里居然把管道输出来了，并且是以，感觉有点儿秀的样子呢
tpot.export('tpot_mnist_pipeline.py')
"""

#这下面的是没有split的版本的代码
skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)

tpot = TPOTClassifier(generations=20, population_size=100, verbosity=2,
                      cv=skf, config_dict='TPOT light')
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_mnist_pipeline.py')
tpot.fit(X_train_scaled, Y_train)
print(tpot.score(X_train_scaled, Y_train))
tpot.export('tpot_mnist_pipeline.py')