#coding=utf-8
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
#将X_train与X_test写入到文件中查看是否存在异样，现在两个对象内的数据明显是一样的呀
#pd.DataFrame(data=X_train).to_csv("C:\\Users\\win7\\Desktop\\X_train.csv", index=False)
#pd.DataFrame(data=X_test).to_csv("C:\\Users\\win7\\Desktop\\X_test.csv", index=False)
#原来这样的写法是横向拼接，X_test只有400多个和891个数据的X_train拼接，后面400多数据必然出现NAN
#X_all = pd.concat([X_train, X_test], axis=1)
#pd.DataFrame(data=X_all).to_csv("C:\\Users\\win7\\Desktop\\X_all.csv", index=False)
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
        self.dropout = nn.Dropout(0.5)
        
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
            
module1 = MyModule1()
module2 = MyModule2()

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

#skf = StratifiedKFold(Y_split_train, n_folds=20, shuffle=True, random_state=None)

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

#我觉得这个中文文档介绍hyperopt还是比较好https://www.jianshu.com/p/35eed1567463
def nn_f(params):
    
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    #print("optimizer__betas", params["optimizer__betas"])
    print("module", params["module"])
    #X = data_set[0].values.astype(np.float32)
    #Y = data_set[1].values.astype(np.longlong)
    
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
    
    skf = StratifiedKFold(Y_train, n_folds=5, shuffle=True, random_state=None)
    
    #metric = cross_val_score(clf, X, Y, cv=skf, scoring="accuracy").mean()
    metric = cross_val_score(clf, X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()
    return -metric
    
def params_cmp(item1, item2):
    
    if not isclose(item1['result']['loss'], item2['result']['loss']):
        if item1['result']['loss'] < item2['result']['loss']:
            return 1
        else: 
            return -1
    else:
        return 0
    
def print_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    print()
    print("the accuracy rate of the model on the whole train dataset is:", acc)
    print()
        
def print_best_params_acc(trials):
    
    trials_list =[]
    #从trials中读取最大的准确率信息咯
    #item和result其实指向了一个dict对象
    for item in trials.trials:
        trials_list.append(item)
    
    #按照关键词进行排序，关键词即为item['result']['loss']
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    print("best parameter is:", trials_list[0])
    print()
    
def record_best_model_acc(clf, acc, best_model, best_acc):
    
    if not isclose(best_acc, acc):
        if best_acc < acc:
            best_acc = acc
            best_model = clf
            
    return best_model, best_acc

def parse_space(trials, space_nodes, best_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    #'vals': {'batch_size': [5], 'criterion': [1], 'lr': [0.0002917044295609288], 'module': [6], 'optimizer__betas': [1], 'optimizer__weight_decay': [0.002568822642786528]}}, 
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["lr"] = trials_list[0]["misc"]["vals"]["lr"][0]
    best_nodes["module"] = space_nodes["module"][trials_list[0]["misc"]["vals"]["module"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]
    
    return best_nodes
    
#我觉得衡量一个分类器效果如何最好的指标就是cv的结果如何
#虽然cv会损失一些数据，但是这些数据可以想办法加上去训练
#既然明确了使用cv作为判断的参数那么就准备改造代码咯
def predict(trials, space_nodes, best_nodes, max_evals=10):
    
    params = parse_space(trials, space_nodes, best_nodes)
    
    best_acc = 0.0
    best_model = 0.0
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
        #我大概知道为什么nn_f可以运行但是predict函数无法执行了
        #因为在前者函数中print(clf.module)和后者中输出不一样
        #归根到底应该是domin负责将space中的参数解析了一次的缘故
        metric = cross_val_score(clf, X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        
        best_model, best_acc = record_best_model_acc(clf, metric, best_model, best_acc)
        
        #因为clf并没有进行过fit所以直接调用下面的函数会出问题
        #print_nnclf_acc(clf, X_train_scaled, Y_train)
        #所以现在不调用函数了直接print结果就可以咯
        print()
        print("the accuracy rate of the classifier on the train dataset is:", metric)
        print()

    #前面只使用cv的方式评价一个较好的模型，但是由于有部分数据未参加训练
    #所以现在在对已经训练好的模型，在进行一次拟合希望能够弥补部分数据的缺少训练
    #用best_model对于拟合模型在进行一次计算咯
    best_model.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 

    print_nnclf_acc(best_model, X_train_scaled, Y_train)
    
    Y_pred = best_model.predict(X_test.values.astype(np.float32))
    #将得到的预测结果写入到文件中去咯
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
    output.to_csv("C:\\Users\\1\\Desktop\\Titanic_Prediction.csv", index=False)
    print("prediction file has been written.")

"""
def predict(trials, space_nodes, best_nodes, max_evals=10):
    
    params = parse_space(trials, space_nodes, best_nodes)
    
    #Python中的变量定义时候就必须初始化的么，大概是因为引用的缘故吧
    best_model = 0.0
    best_accuracy = 0.0
    
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
    
        #我大概知道为什么nn_f可以运行但是predict函数无法执行了
        #因为在前者函数中print(clf.module)和后者中输出不一样
        #归根到底应该是domin负责将space中的参数解析了一次的缘故
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong))
        
        Y_pred = clf.predict(X_train_scaled.values.astype(np.float32))
        #count = (Y_pred == (Y_train.values)).sum()
        count = (Y_pred == Y_train).sum()
        acc = count/len(Y_train)
        print()
        print("the accuracy rate on the whole train dataset is:", acc)
        print()
    
    #需要注意的是X_test和X_train应该一起归一化吧
    #然后显示该模型在现在的数据集上的拟合情况
    Y_pred = clf.predict(X_test.values.astype(np.float32))
    #将得到的预测结果写入到文件中去咯
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
    output.to_csv("C:\\Users\\win7\\Desktop\\Titanic_Prediction.csv", index=False)
"""

"""
params = {
    "max_epochs":[40, 60, 80, 100, 120],#, 120, 200, 240],     
    "lr": [0.0001, 0.0002, 0.0005, 0.001],#, 0.002, 0.005, 0.01, 0.02, 0.05],
    "optimizer__weight_decay":[0, 0.001, 0.002, 0.005, 0.01],
    "criterion":[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
    "batch_size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "optimizer__betas":[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], [0.94, 0.999],
        [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]],
    "module":[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
              module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
}
"""

space = {"lr":hp.uniform("lr", 0.0001, 0.0010),  
         "optimizer__weight_decay":hp.uniform("optimizer__weight_decay", 0, 0.01),  
         "criterion":hp.choice("criterion", [torch.nn.NLLLoss, torch.nn.CrossEntropyLoss]),
         #batchsize或许可以改为hp.randint，哦并不能够修改为randint否则可能取0哈，用hp.qloguniform(label, low, high, q)q取1
         "batch_size":hp.choice("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
         "optimizer__betas":hp.choice("optimizer__betas",[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], 
         [0.94, 0.999], [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]]),
         "module":hp.choice("module", [module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
               module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]),
         }

space_nodes = {"lr":[0.0001],
               "optimizer__weight_decay":[0.005],
               "criterion":[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
               "batch_size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
               "optimizer__betas":[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], [0.94, 0.999],
                       [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]],
               "module":[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
                     module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
               }

best_nodes = {"lr":0.0001,
              "optimizer__weight_decay":0.005,
              "criterion":torch.nn.NLLLoss,
              "batch_size":1,
              "optimizer__betas":[0.86, 0.999],
              "module":module1.weight_init1(),
             }

trials = Trials()
#TPE即为Tree of Parzen Estimators，也就是贝叶斯优化的一个变体
#这个库也提供了模拟退火(对应是hyperopt.anneal.suggest) 
#我之前还想用algo = partial(bayes.suggest, n_startup_jobs=10)
#让上面的语句报错，这样我就知道partial的参数可以是什么类型了
#结果最后Google了一下TPE就是贝叶斯优化的变种，之前的中文博客文档啊写错了
algo = partial(tpe.suggest, n_startup_jobs=10)
#dict = {X_train:X_train_scaled, Y_train:Y_train}
#data_set = [X_train_scaled, Y_train]
#原来fmin(self, fn, space, **kw):函数列表是这样的algo是kw.get得到的
#果然在执行fmin中的nn_f之前会先计算几个点，然后才进行迭代操作
#贝叶斯优化算法的原理确实如此，tpe也是一种贝叶斯优化变形
best_params = fmin(nn_f, space, algo=algo, max_evals=10, trials=trials)
print_best_params_acc(trials)
#如果只是记录下这组超参的话，下面的办法可以实现记录，再用下面的超参搜索咯，毕竟CV不够好
predict(trials, space_nodes, best_nodes, max_evals=10)
#print(hyperopt.space_eval(space, best_params))
#predict(best_params)
"""
#我的这个超参搜索其实主要针对于lr、criterion、module（所选择的的模型已经模型采用的初始化方式）还有optimizer__weight_decay
#如果将cv的值设置的更大一些，准确率应该会更准确一些的，但是已经能够反应参数效果了所以建议将随机超参搜索的cv值也设置小一点
#我觉得因为神经网络的参数空间巨大而且存在很多随机设置所以导致超参搜索的效果并不是特别的理想，但是能搜索的数据集尽量还是搜索
#迄今为止我对这个超参搜索库的理解：如果仅仅很有限的计算资源随机超参搜索可能效果好，但是如果有较多的资源那么这个库搜索结果更好
print()
#讲真，我不是太看得懂trials里面存储的东西
#下面的数据其实是以BSON的格式存储的训练信息
#然后下面的trials[:2]信息其实就是trial的前两项数据（包含loss信息）
#for item in trials.trials: 这个显示结果和上面差不多呀
for item in trials.trials[:2]:
    print(item)
print()
print(best)
"""
#现在准备解决的问题：
#（1）加入预测的元素
#（2）加入噪声的元素
#（3）查看逻辑回归的fit是否只有一个epoch
#（4）然后使用hpsklearn的库进行训练咯，一切以sklearn为核心
#（5）提交Titanic的结果咯
#（6）然后形成模块3咯
#（7）准备查找特征工程的部分形成模块4咯