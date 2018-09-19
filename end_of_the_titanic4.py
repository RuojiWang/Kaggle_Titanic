#coding=utf-8
#我之前试运行了一下这个版本，感觉还是有一些可以改进的空间
#（1）缺少运行进度的显示，我经常不知道还要多久才能算完
#（2）缺少异常恢复机制（或者中间结果存储机制），他妈的一断电所有的计算都白做了。
#（3）每次运行程序之前需要先确认程序能够跑通再大量的计算咯（文件路径、cuda容易出错）
#（4）然后这么大的计算量，并没有带来预期的巨大提升，网络结构的搜索肯定也会这样
#进度条功能是可以实现的、异常恢复太难了只能够存储中间结果、顺便试一下TPOT正确率？
#我感觉可能没办法在运行时备份的吧，这个需要特殊的软硬件才能实现服务器实时备份的吧

#我尼玛发生了一件奇怪的事情，使用cpu的速度居然比gpu速度更快。
#2018-9-9 14:12:32这个版本算是一个里程碑版本吧，我要在这个版本基础上开发新的版本
import sys
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

from skorch import NeuralNetClassifier
import sys
import skorch
import torch.nn.init
#引入这个东西主要是为了查看MLPClassifier的属性集吧
from sklearn.neural_network import MLPClassifier
from sympy.polys.tests.test_ring_series import is_close

import pickle
from nltk.draw.util import SpaceWidget
from end_of_the_titanic1 import MyModule1

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

import sys
sys.path.append("D:\\Workspace\\Titanic")
from Utilities1 import noise_augment_pytorch_classifier

class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()

        self.fc1 = nn.Linear(9, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def weight_init1(self):
        return self
        
    def weight_init2(self):
        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.normal_(self.fc3.weight)
        torch.nn.init.constant_(self.fc3.bias, 0)
        return self
    
    def weight_init3(self):
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.constant_(self.fc3.bias, 0)
        return self
        
    def weight_init4(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        return self
    
class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()

        self.fc1 = nn.Linear(9, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = self.dropout2(X)
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def weight_init1(self):
        return self
        
    def weight_init2(self):
        torch.nn.init.normal_(self.fc1.weight.data)
        torch.nn.init.constant_(self.fc1.bias.data, 0)
        torch.nn.init.normal_(self.fc2.weight.data)
        torch.nn.init.constant_(self.fc2.bias.data, 0)
        torch.nn.init.normal_(self.fc3.weight.data)
        torch.nn.init.constant_(self.fc3.bias.data, 0)
        torch.nn.init.normal_(self.fc4.weight.data)
        torch.nn.init.constant_(self.fc4.bias.data, 0)
        return self
    
    def weight_init3(self):
        torch.nn.init.xavier_normal_(self.fc1.weight.data)
        torch.nn.init.constant_(self.fc1.bias.data, 0)
        torch.nn.init.xavier_normal_(self.fc2.weight.data)
        torch.nn.init.constant_(self.fc2.bias.data, 0)
        torch.nn.init.xavier_normal_(self.fc3.weight.data)
        torch.nn.init.constant_(self.fc3.bias.data, 0)
        torch.nn.init.xavier_normal_(self.fc4.weight.data)
        torch.nn.init.constant_(self.fc4.bias.data, 0)
        return self
    
    def weight_init4(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        return self   
            
class MyModule3(nn.Module):
    def __init__(self):
        super(MyModule3, self).__init__()

        self.fc1 = nn.Linear(9, 20)
        self.fc2 = nn.Linear(20, 40)
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

    def weight_init1(self):
        return self
        
    def weight_init2(self):
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
        return self
    
    def weight_init3(self):
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
        return self
    
    def weight_init4(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        return self   
    
class MyModule4(nn.Module):
    def __init__(self):
        super(MyModule4, self).__init__()

        self.fc1 = nn.Linear(9, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 40)
        self.fc7 = nn.Linear(40, 40)
        self.fc8 = nn.Linear(40, 40)
        self.fc9 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = F.relu(self.fc5(X))
        X = self.dropout1(X)
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.relu(self.fc8(X))
        X = F.softmax(self.fc9(X), dim=-1)
        return X

    def weight_init1(self):
        return self
        
    def weight_init2(self):
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
        torch.nn.init.normal_(self.fc9.weight.data)
        torch.nn.init.constant_(self.fc9.bias.data, 0)
        return self
    
    def weight_init3(self):
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
        torch.nn.init.xavier_normal_(self.fc9.weight.data)
        torch.nn.init.constant_(self.fc9.bias.data, 0)
        return self
    
    def weight_init4(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        torch.nn.init.xavier_uniform_(self.fc6.weight.data)
        torch.nn.init.xavier_uniform_(self.fc7.weight.data)
        torch.nn.init.xavier_uniform_(self.fc8.weight.data)
        torch.nn.init.xavier_uniform_(self.fc9.weight.data)
        return self   
    
module1 = MyModule1()
module2 = MyModule2()
module3 = MyModule3()
module4 = MyModule4()

net = NeuralNetClassifier(
    module = module1,
    lr=0.1,
    #device="cuda",
    device="cpu",
    max_epochs=80,
    #criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[skorch.callbacks.EarlyStopping(patience=8)]
)

X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.01, shuffle=True)


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pickle
import time
import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc

def print_nnclf_acc(acc):
    
    print("the accuracy rate of the model on the whole train dataset is:", acc)
  
def print_best_params_acc(trials):
    
    trials_list =[]
    #从trials中读取最大的准确率信息咯
    #item和result其实指向了一个dict对象
    for item in trials.trials:
        trials_list.append(item)
    
    #按照关键词进行排序，关键词即为item['result']['loss']
    trials_list.sort(key=lambda item: item["result"]["loss"])
    
    print("best parameter is:", trials_list[0])
    print()
    
def save_inter_params(trials, space_nodes, best_nodes):
 
    files = open("titanic_intermediate_parameters.pickle", "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params():
  
    files = open("titanic_intermediate_parameters.pickle", "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes
    
def save_best_model(best_model):
    
    files = open("titanic_best_model.pickle", "wb")
    pickle.dump(best_model, files)
    files.close()
    
def load_best_model():
    
    files = open("titanic_intermediate_parameters.pickle", "rb")
    best_model = pickle.load(files)
    files.close()
    
    return best_model
    
def record_best_model_acc(clf, acc, best_model, best_acc):
    
    flag = False
    
    if not isclose(best_acc, acc):
        if best_acc < acc:
            flag = True
            best_acc = acc
            best_model = clf
            
    return best_model, best_acc, flag

def noise_augment_data(mean, std, X_train, Y_train, columns):
    
    #这样的设置应该不是修改X_train的数据咯
    #这里可以通过id(X_train)查看两个变量地址
    #用这样的方式去判断两个变量是否对应同一个对象简单多了
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    #获取数据的行数目，并对每一行中colums中的列添加噪声
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] +=  random.gauss(mean, std)

    return X_noise_train, Y_train
    
#我觉得这个中文文档介绍hyperopt还是比较好https://www.jianshu.com/p/35eed1567463
def nn_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("module", params["module"])    
    
    #X_noise_train与X_train_scaled不同（非同一份拷贝），但是Y_noise_train与Y_train相同
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              module=params["module"],
                              #下面都是固定的参数咯
                              #device="cuda",
                              device="cpu",
                              #我就说为毛每次都是计算十次呢，才想到是这里clf用了默认参数的缘故
                              max_epochs = 400,
                              optimizer=torch.optim.Adam,
                              callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
                              )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric

#将display_search_progress代替nn_f传入参数好像没啥卵用
def display_search_progress(search_times, nn_f):
    
    print("search times:", search_times)
    return nn_f
    
def parse_space(trials, space_nodes, best_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    #'vals': {'batch_size': [5], 'criterion': [1], 'lr': [0.0002917044295609288], 'module': [6], 'optimizer__betas': [1], 'optimizer__weight_decay': [0.002568822642786528]}}, 
    #best_nodes["mean"] = trials_list[0]["misc"]["vals"]["mean"][0]
    #best_nodes["std"] = trials_list[0]["misc"]["vals"]["std"][0]
    #卧槽我走查真的发现了一个错误..我之前还不情愿，认为检查过函数接口就行了
    #不过这个错误并不影响结果就是了..因为预测的时候并没有用到这两个参数咯
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["lr"] = trials_list[0]["misc"]["vals"]["lr"][0]
    best_nodes["module"] = space_nodes["module"][trials_list[0]["misc"]["vals"]["module"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]
    
    return best_nodes
    
def predict(best_nodes, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0
    
    #在预测之前好像并不需要使用数据集增强，不然感觉出现了奇怪的问题咯
    #X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module=best_nodes["module"],
                              
                                  #下面都是固定的参数咯
                                  #device="cuda",
                                  device="cpu",
                                  #我就说为毛每次都是计算十次呢，才想到是这里clf用了默认参数的缘故
                                  #改了新模型我总觉得似乎不太满意还是将最大训练值给修改了吧
                                  max_epochs = 400,
                                  optimizer=torch.optim.Adam,
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
                                  )
     
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        #尼玛这里面不能够进行单步调试不然就会报错咯
        #尼玛下面的被注释的写法造成了进入if语句会报错，真的神奇呀
        #if flag
        if (flag):        
            
            save_best_model(best_model)
            Y_pred = best_model.predict(X_test_scaled.values.astype(np.float32))
            #将得到的预测结果写入到文件中去咯
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            output.to_csv("C:/Users/win7/Desktop/Titanic_Prediction.csv", index=False)
            print("prediction file has been written.")
        print()
        
    
    #输出最佳模型的正确率等情况咯
    metric = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
    print("the best accuracy rate of the model on the whole train dataset is:", metric)
    
"""
def predict(trials, space_nodes, best_nodes, max_evals=10):
    
    params = parse_space(trials, space_nodes, best_nodes)
    
    best_acc = 0.0
    best_model = 0.0
    
    #在预测之前好像并不需要使用数据集增强，不然感觉出现了奇怪的问题咯
    #X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    
    for i in range(0, max_evals):
        
        clf = NeuralNetClassifier(lr = params["lr"],
                                  optimizer__weight_decay = params["optimizer__weight_decay"],
                                  criterion = params["criterion"],
                                  batch_size = params["batch_size"],
                                  optimizer__betas = params["optimizer__betas"],
                                  module=params["module"],
                              
                                  #下面都是固定的参数咯
                                  #device="cuda",
                                  device="cpu",
                                  #我就说为毛每次都是计算十次呢，才想到是这里clf用了默认参数的缘故
                                  max_epochs = 200,
                                  optimizer=torch.optim.Adam,
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
                                  )
     
        skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)
        
        metric = cross_val_score(clf, X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        
        best_model, best_acc = record_best_model_acc(clf, metric, best_model, best_acc)
        
        print()
        print("the accuracy rate of the classifier on the train dataset is:", metric)
        print()

    best_model.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 

    print_nnclf_acc(best_model, X_train_scaled, Y_train)
    
    Y_pred = best_model.predict(X_test.values.astype(np.float32))
    #将得到的预测结果写入到文件中去咯
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
    output.to_csv("C:\\Users\\1\\Desktop\\Titanic_Prediction.csv", index=False)
    print("prediction file has been written.")
"""

space = {"mean":hp.choice("mean", [0]),
         #"std":hp.choice("std", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         #                        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
         "std":hp.choice("std", [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
                                 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]),
         "lr":hp.uniform("lr", 0.0001, 0.0010),  
         "optimizer__weight_decay":hp.uniform("optimizer__weight_decay", 0, 0.01),  
         "criterion":hp.choice("criterion", [torch.nn.NLLLoss, torch.nn.CrossEntropyLoss]),
         #batchsize或许可以改为hp.randint，哦并不能够修改为randint否则可能取0哈，用hp.qloguniform(label, low, high, q)q取1
         "batch_size":hp.choice("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
         "optimizer__betas":hp.choice("optimizer__betas",[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], 
         [0.94, 0.999], [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]]),
         #发生了一件奇怪的事情咯，我用moudle3和module4训练出来的用于训练module1和module2居然似乎取得了更好的结果
         #还好老子一直盯着屏幕再看，才发现了这个问题，果然是一处的修改都会涉及到很多的其他位置的修改毕竟是超参搜索
         #"module":hp.choice("module", [module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
         #      module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()])
         "module":hp.choice("module", [module3.weight_init1(), module3.weight_init2(), module3.weight_init3(), module3.weight_init4(), 
               module4.weight_init1(), module4.weight_init2(), module4.weight_init3(), module4.weight_init4()])         
         }

space_nodes = {"mean":[0],
               "std":[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
                     0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40],
               "lr":[0.0001],
               "optimizer__weight_decay":[0.005],
               "criterion":[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
               "batch_size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
               "optimizer__betas":[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], [0.94, 0.999],
                       [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]],
               #"module":[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
               #      module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
               #module1和module2计算一天的上线也就是84.5%的准确率了吧，被TPOT完爆了，感觉还是需要构造新模型咯所以有了下面的module3和module4咯
               #就这个数据集而言哈，我觉得无脑增加模型的层数似乎就可以增加最终拟合的效果了耶。。
               "module":[module3.weight_init1(), module3.weight_init2(), module3.weight_init3(), module3.weight_init4(), 
                     module4.weight_init1(), module4.weight_init2(), module4.weight_init3(), module4.weight_init4()]
               }

best_nodes = {"mean":0,
              "std":0.1,
              "lr":0.0001,
              "optimizer__weight_decay":0.005,
              "criterion":torch.nn.NLLLoss,
              "batch_size":1,
              "optimizer__betas":[0.86, 0.999],
              "module":module3.weight_init1(),
             }

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=1, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_space(trials, space_nodes, best_nodes)
save_inter_params(trials, space_nodes, best_nodes)
trials, space_nodes, best_nodes = load_inter_params()
#因为神经网络的训练采用了early stopping，所以不是很怕过拟合的问题
#在最终预测之前，对于最优的模型还进行了一次全数据的训练，可能不是最好但是比较靠谱
#我随便运行了一下程序，很小的计算量下面也能得到比较靠谱的结果
#另外，这个最优模型是指交叉验证某一回合中最优的分类器，感觉好像也只能这么算咯
#这个其实还可以进行一点优化，貌似直接用整个数据集对于模型进行训练更靠谱吧？
#当你觉得某种做法比较奇怪的时候，或者是觉得程序很难写的时候，一定是哪里出了问题。。
#之前之所以在predict中执行奇怪的操作主要就是没想到其实神经网络已经采用了early-stopping..
predict(best_nodes, max_evals=50)
#现在我发现了两个技术问题哈，其实本质是上一个问题出现在两个位置
#也就是模型似乎是使用的同一个模型咯，我觉得不应该出现这种状况吧
#cv的时候使用的是已经训练过的模型？预测的时候也是使用的同样的模型再训练
#中间的计算结果还是被保存下来了的，但是cpu的计算速度似乎真的比GPU快呀
#params["module"]是指向了初始化之后的module
#意思就是说MyModule3和MyModule4各自包含有4个已经初始化的module
#在进行贝叶斯优化的时候仅仅从这8个初始化起始位置开始搜索最优超参
#怎么证明这个说法呢，证明的过程其实很简单就是nn_f中输出id(params["module"])就完事儿了
#cross_val_score其实也没问题就是在8个初始化起始位置开始进行交叉验证选择模型
#然后predict的时候直接将之前贝叶斯优化训练好的模型再训练，所以训练特别的快但是结果都差不多
#我觉得解决这个问题的办法就是直接每次都重新生成模型就完事儿了吧，不同的起始位置对于神经网络还是很重要的