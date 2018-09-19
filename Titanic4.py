#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

def VarianceThreshold_selector(data, threshold):
    columns = data.columns
    selector = VarianceThreshold(threshold)
    selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True)]
    feature = pd.DataFrame(selector.fit_transform(data), columns=labels)
    return feature

def RFE_selector(estimator, n_features_to_select, X_data, Y_data):
    columns = X_data.columns
    selector = RFE(estimator = estimator, n_features_to_select = n_features_to_select)
    selector.fit_transform(X_data, Y_data)
    labels = [columns[x] for x in selector.get_support(indices=True)]    
    feature = pd.DataFrame(selector.fit_transform(X_data, Y_data), columns=labels)
    return feature

def SelectFromModel_selector(estimator, threshold, X_data, Y_data):
    columns = X_data.columns
    selector = SelectFromModel(estimator, threshold = threshold)
    selector.fit_transform(X_data, Y_data)
    labels = [columns[x] for x in selector.get_support(indices=True)]    
    feature = pd.DataFrame(selector.fit_transform(X_data, Y_data), columns=labels)
    return feature

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def compare_score(score, battle, n):    
    index = 0
    max_score = 0.0
    for i in range(0, n):
        if(not isclose(max_score, score[i])):
            if(max_score < score[i]):
                max_score = score[i]
                index = i
    
    for i in range(0, n):
        if(i==index or isclose(score[i], max_score)):
            battle[i] += 1
    
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
    
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

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
X_train_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)

#以后只有非常重要的东西才两边都传递吧，不然一直单边存在文件就行了。
#根据二八定律这种文件存在的极少所以不用特别担心呢。以后文件不再传递Titanic吧。

#这个原来版本的代码怎么调试都有点问题，算了感觉还是自己写一个吧，比调试学到的东西多
x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(X_train_scaled, Y_train, test_size = 0.1)
x_train = x_train_df.values
x_val = x_val_df.values
y_train = y_train_df.values
y_val = y_val_df.values

#模型定义相关的代码咯
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 270)
        self.fc2 = nn.Linear(270, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        
        return x
        
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 100

#这个代码好像没得改呀，改了之后enumerate内必须使用dataloader
#然而使用dataloader的话代码的其他处理部分也必须类似Titan3的getitem。。
#所以我觉得这个写到这个成都就差不多了，再改要类似Titanic3代码而且没它规范
for epoch in range(epochs):
    for index, inputs in enumerate(x_train):
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    # get accuracy
    correct_num = 0
    for index, (inputs, targets) in enumerate(validation_data_loader):
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs).data.numpy()
        predicted = np.where(outputs > 0.5, 1, 0)
        answer = targets.data.numpy()
        correct_num += (predicted == answer).sum()
    print('Epoch [%d/%d], Loss:%.4f, Accuracy:%.4f' % (epoch+1, epochs, loss.data[0], correct_num/len(validation_set)))
    # generate_kaggle_data(model)
    