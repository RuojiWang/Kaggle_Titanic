#coding=utf-8
#这个系列的代码主要就是用于尝试迁移学习咯，我试一下titanic的东西如何进行迁移学习
#具体的资料先看看这个再说咯 https://cloud.tencent.com/developer/article/1495081
#假如两个领域之间的区别特别的大，不可以直接采用迁移学习，因为在这种情况下效果不是很好。
#在这种情况下，推荐以上的方法，在两个相似度很低的domain之间一步步迁移过去（踩着石头过河）。
#目前俺们村儿的titanic就是属于相似性差很多的情况，如果迁移学习恐怕也是一步一步的迁移过去，裂开了。。
#以后做比赛俺需要用到 https://www.jianshu.com/p/59cdbf7439df 这里面提到的fastai咯
#这里有Pytorch自带的迁移学习相关模块：
#https://pytorch.org/docs/0.3.0/torchvision/models.html和https://www.jianshu.com/p/d04c17368922
#torch.nn.NLLLoss torch.nn.CrossEntropyLoss nn.Softmax 这三个好像经常容易出现各种错误吧
#https://www.pytorchtutorial.com/mofan-pytorch-tutorials-list/ 这个是莫凡的回归网络和分类网络的例子


#之后的所有transferlearning都直接使用 autogluon来做嘛。不过这个库目前只支持Linux和苹果的系统，
#可能以后会支持Windows把，也许我花在Linux上面的时间也不算白费
#你敢相信这个autogluon的效果在原始数据集上面居然比我经过了处理的数据集上面效果更好


import os
import sys
import math
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

# 原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split


import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK


# 下面的这个kfold是实现k折交叉的功能，返回每次的indice，可以设置为shuffle但默认未设
# 然后这个StratifiedKFold是返回k折交叉的迭代器，每次通过迭代器返回结果，可以设置为shuffle
# 两者的区别在于前者返回indice或者索引列表后者直接返回迭代器，虽然我这一份代码两种方式都有但是让他们并存吧
# from sklearn.model_selection import KFold,StratifiedKFold

# 因为每次都出现警告显得很麻烦所以用下面的方式忽略
# ignore warning from packages
warnings.filterwarnings('ignore')

# load train data and test data
data_train = pd.read_csv("kaggle_titanic_files/train.csv")
data_test = pd.read_csv("kaggle_titanic_files/test.csv")
# combine train data and test data
combine = [data_train, data_test]

# create title feature from name feature
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Capt'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# create familysize from sibsp and parch feature
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

# 现在这里的性别不能够被替换否则下面无法使用DictVectorizer进行映射
# switch sex to number, which could make following missing age prediction easier
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# predict missing age
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]
    dataset['Age'] = dataset['Age'].astype(int)

# 为了之后使用dictvector进行映射，现在又将性别变为字符串
# 如果不用这样的操作变为字符串就无法用dictvector替换啦
# 因为dictvector的替换就是直接将非数值的东西变成单独的一类
# switch sex to string for later DictVectorizer
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({1: 'female', 0: 'male'})

# 这里的mode是求解pandas.core.series.Series众数的第一个值（可能有多个众数）
# fill missing data in embarked feature
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 将data_test中的fare元素所缺失的部分由已经包含的数据的中位数决定哈
# fill missing data in fare
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

# switch cabin feature into "cabin" or "no cabin"
for dataset in combine:
    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = "cabin"
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = "no cabin"

# switch number of pclass into string for later DictVectorizer
for dataset in combine:
    dataset.loc[dataset['Pclass'] == 1, 'Pclass'] = "1st"
    dataset.loc[dataset['Pclass'] == 2, 'Pclass'] = "2nd"
    dataset.loc[dataset['Pclass'] == 3, 'Pclass'] = "3rd"

# 尼玛给你说的这个是共享船票，原来的英文里面根本就没有这种说法嘛
# find if there is the same ticket, which means higher probability of survival
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
# print(df)
df_ticket = df.index.values  # 共享船票的票号
tickets = data_train.Ticket.values  # 所有的船票
# print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0  # 遍历所有船票，在共享船票里面的为1，否则为0
    result.append(ticket)
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values  # 共享船票的票号
tickets = data_train.Ticket.values  # 所有的船票

# create ticket_count feature in training data
result = []
for ticket in tickets:
    if ticket in df_ticket:
        # 主要是为了使用DictVectorizer类映射所以改写下面的样子
        # ticket = 1
        ticket = "share"
    else:
        # ticket = 0                   #遍历所有船票，在共享船票里面的为1，否则为0
        ticket = "no share"
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_train = pd.concat([data_train, results], axis=1)

# create ticket_count feature in test data
df = data_test['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values
tickets = data_test.Ticket.values
result = []
for ticket in tickets:
    if ticket in df_ticket:
        # ticket = 1
        ticket = "share"
    else:
        # ticket = 0
        ticket = "no share"
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_test = pd.concat([data_test, results], axis=1)

# data_train_1 data_train data_test_1 data_test have different id
data_train_1 = data_train.copy()
data_test_1 = data_test.copy()
data_test_1 = data_test_1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1)

# get feature from data_train_1 and data_test_1
X_train = data_train_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]
Y_train = data_train_1['Survived']
X_test = data_test_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]

# combine X_train and X_test
X_all = pd.concat([X_train, X_test], axis=0)
# print(X_all.columns)
# 下面是我补充的将性别、姓名、Embarked修改为了one-hot编码类型了
# 原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
# use DictVectorizer to create something like one-hot encoding
dict_vector = DictVectorizer(sparse=False)
X_all = dict_vector.fit_transform(X_all.to_dict(orient='record'))
X_all = pd.DataFrame(data=X_all, columns=dict_vector.feature_names_)

# scale feature to (0,1), which could make process of training easier
X_all_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns=X_all.columns)
X_train_scaled = X_all_scaled[:len(X_train)] #因为训练数据总数为891将其改变为880，以避免十折交叉验证的时候batchnorm的batchsize为1
Y_train = Y_train[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]



# 后对数据进行拼接咯
X_train_scaled = pd.concat([X_train_scaled, Y_train], axis=1)



import pandas as pd
from autogluon import TabularPrediction as task

#train_data = task.Dataset(file_path="kaggle_titanic_files/train.csv")
#test_data = task.Dataset(file_path="kaggle_titanic_files/test.csv")
#predictor = task.fit(train_data=train_data, label='Survived')
#y_pred = predictor.predict(test_data)

predictor = task.fit(train_data=X_train_scaled, label='Survived')
y_pred = predictor.predict(X_test_scaled)

sub_csv = pd.DataFrame({'PassengerId': data_test["PassengerId"], 'Survived': y_pred})
sub_csv.to_csv("kaggle_titanic_files/autogluon_submission_scaled.csv", index=False)

sub = pd.read_csv("kaggle_titanic_files/autogluon_submission_scaled.csv")
sub['Survived'] = y_pred