#coding=utf-8
#下面的代码主要是2018-9-17计算10000/1000次的分析
#创建这样一个文件也便于分析存储神经网络的结构
#毕竟每次大计算机我都会修改神经网络的结构
#这次我在原代码上面添加了计时功能显示运算了28个小时。。
#我觉得现在我现在大致的情况是：
#我现在对这个问题的理解，除了网络结构以外的超参都调的差不多了。
#但是结果不是很理想呀，正在郁闷不知道咋样处理呢。。
#我觉得我现在面临的问题主要是：
#（1）我不知道两次大计算算出来的模型哪个更好？
#针对这个问题，下次选择好最佳超参的时候，用最佳超参训练模型
#然后将模型训练完之后在进行一次交叉验证码？但是训练完成之后
#在进行交叉验证还有意义吗，必须是没使用过的数据才能用于验证吧
#所以目前看来这个问题还是比较玄学的咯。。
#（2）然后我不知道怎么对于模型进行进一步的优化。
#以我有限的知识和经验似乎模型层面的优化似乎提升大概也就是这么多了吧
#现在的我好想才是真的不知所措了吧
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
sys.path.append("D:\\Workspace\\Titanic")
from Utilities1 import noise_augment_pytorch_classifier

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import skorch
from skorch import NeuralNetClassifier

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc

data_train = pd.read_csv("C:/Users/1/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/1/Desktop/test.csv")
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

class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc1.bias.data, 0)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self   
    
class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 
    
class MyModule3(nn.Module):
    def __init__(self):
        super(MyModule3, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 

class MyModule4(nn.Module):
    def __init__(self):
        super(MyModule4, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 

class MyModule5(nn.Module):
    def __init__(self):
        super(MyModule5, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self   
    
class MyModule6(nn.Module):
    def __init__(self):
        super(MyModule6, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self
    
class MyModule7(nn.Module):
    def __init__(self):
        super(MyModule7, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self

class MyModule8(nn.Module):
    def __init__(self):
        super(MyModule8, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self

class MyModule9(nn.Module):
    def __init__(self):
        super(MyModule9, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self   
    
            
class MyModule10(nn.Module):
    def __init__(self):
        super(MyModule10, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self

class MyModule11(nn.Module):
    def __init__(self):
        super(MyModule11, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self

class MyModule12(nn.Module):
    def __init__(self):
        super(MyModule12, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#就是使用默认设置的意思咯
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self
       
class MyModule13(nn.Module):
    def __init__(self):
        super(MyModule13, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule14(nn.Module):
    def __init__(self):
        super(MyModule14, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self

class MyModule15(nn.Module):
    def __init__(self):
        super(MyModule15, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule16(nn.Module):
    def __init__(self):
        super(MyModule16, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule17(nn.Module):
    def __init__(self):
        super(MyModule17, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 40)
        self.fc7 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
    
class MyModule18(nn.Module):
    def __init__(self):
        super(MyModule18, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self

class MyModule19(nn.Module):
    def __init__(self):
        super(MyModule19, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 60)
        self.fc7 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
        
class MyModule20(nn.Module):
    def __init__(self):
        super(MyModule20, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 70)
        self.fc7 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
    
class MyModule21(nn.Module):
    def __init__(self):
        super(MyModule21, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 40)
        self.fc7 = nn.Linear(40, 40)
        self.fc8 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule22(nn.Module):
    def __init__(self):
        super(MyModule22, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule23(nn.Module):
    def __init__(self):
        super(MyModule23, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 60)
        self.fc7 = nn.Linear(60, 60)
        self.fc8 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule24(nn.Module):
    def __init__(self):
        super(MyModule24, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 70)
        self.fc7 = nn.Linear(70, 70)
        self.fc8 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#就是什么都不做的意思
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
module1 = MyModule1()
module2 = MyModule2()    
module3 = MyModule3()
module4 = MyModule4()
module5 = MyModule5()
module6 = MyModule6()
module7 = MyModule7()
module8 = MyModule8()
module9 = MyModule9()
module10 = MyModule10()    
module11 = MyModule11()
module12 = MyModule12()
module13 = MyModule13()
module14 = MyModule14()
module15 = MyModule15()
module16 = MyModule16()
module17 = MyModule17()
module18 = MyModule18()    
module19 = MyModule19()
module20 = MyModule20()
module21 = MyModule21()
module22 = MyModule22()
module23 = MyModule23()
module24 = MyModule24()

net = NeuralNetClassifier(
    module = module6,
    lr=0.1,
    #device="cuda",
    device="cpu",
    max_epochs=400,
    #criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[skorch.callbacks.EarlyStopping(patience=10)]
)

files = open("titanic_intermediate_parameters_2018-9-18083418.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
print(best_nodes)
#print(space_nodes)
print()

files = open("titanic_best_model_2018-9-18083508.pickle", "rb")
best_model = pickle.load(files)
files.close()
best_acc = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
print(best_acc)

#输出的居然是下面的结果让我觉得大吃一惊呢，难道是因为随机性？
#按照我的预期，至少在模型选择层面我觉得应该选择五层左右的吧
#但是还是有一些最优参数的结果和上次的比如说是patience以及batch_size咯
#我觉得在参数优化层面可以做点改动就是将optimizer__weight_decay改为固定取值吧
#我觉得也许是不是说明超参选择以及模型结构方面的优化大致也就只能到这个程度了？
#需要我增加类似数据集增强等的东西才能够更好的battle吧？感觉模型结构对于结果影响没那么大
#{'title': 'titanic', 'path': 'path', 'mean': 0, 'std': 0.12, 'max_epochs': 400, 'patience': 9, 'lr': 0.00030619694480391203, 'optimizer__weight_decay': 0.004458260342599436, 'criterion': <class 'torch.nn.modules.loss.NLLLoss'>, 'batch_size': 128, 'optimizer__betas': [0.88, 0.9993], 'module': MyModule3(
#  (fc1): Linear(in_features=9, out_features=60, bias=True)
#  (fc2): Linear(in_features=60, out_features=60, bias=True)
#  (fc3): Linear(in_features=60, out_features=2, bias=True)
#  (dropout1): Dropout(p=0.1)
#  (dropout2): Dropout(p=0.2)
#), 'init_mode': 3, 'device': 'cpu', 'optimizer': <class 'torch.optim.adam.Adam'>}
#
#0.8619528619528619