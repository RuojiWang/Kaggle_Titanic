#coding=utf-8
#这份代码存在的意义主要是测试是否需要split数据集哈，感觉多半不需要的毕竟有early stopping
#还好我一直就有备份每个版本代码的习惯，否则改来改去会花费大量的时间咯
#我觉得应该不需要split数据吧，原因大致如下：
#（1）迄今为止我还未见过仅用split数据进行预测的呢。
#（2）已经有early stopping的方式防止过拟合。
#（3）超参选择时候进行过cv，某种意义上也能够防止模型过拟合。
#（4）你split出数据到底想干什么？也就只能看在未知数据集上表现吧，其实这个模型可以看到位置数据集上表现
#（5）split出来的数据还是会参与最终模型的训练，分开训练效果大概率没有一起训练的效果来的好而且更方便。
#（6）Example6.py其实就是TPOT的自动化模型，TPOT中的CV其实决定了最优的超参而已，并不是找到了最优模型
#所以归根到底，我纠结了这么久就是（1）没搞清楚最优模型和最优超参。（2）没搞清楚split数据用于干嘛。
#另外，更正一个概念：K折交叉验证用于模型调优，找到使得模型泛化性能最优的超参值。
#TPOT其实是选择了最佳超参而已，early stopping、L2正则化等是为了选择最佳模型（超参已经确定的情况下）。
#所以说我的框架四应该上限更高（神经网络模型拟合能力最强，数据集较多训练时间充分时将会取得更优结果），
#但是TPOT应该下限更高（能够在很短的时间内先实现一个原型并提交，如果配合early stopping应该可以达到更好的结果）
#框架四需要在构造模型结构方面花一些时间，TPOT需要在特征工程方面花更多的时间，总体而言可能框架四更具潜力。
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
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

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
    train_test_split(X_train_scaled, Y_train, test_size =0.15, shuffle=True)

#我觉得这个中文文档介绍hyperopt还是比较好https://www.jianshu.com/p/35eed1567463
def nn_f(params):
    
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    #print("optimizer__betas", params["optimizer__betas"])
    print("module", params["module"])
    
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
    
    skf = StratifiedKFold(Y_split_train, n_folds=5, shuffle=True, random_state=None)
    
    metric = cross_val_score(clf, X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
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
     
        skf = StratifiedKFold(Y_split_train, n_folds=5, shuffle=True, random_state=None)

        metric = cross_val_score(clf, X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        
        best_model, best_acc = record_best_model_acc(clf, metric, best_model, best_acc)

        print()
        print("the accuracy rate of the classifier on the train dataset is:", metric)
        print()
        
        #我试了二十分钟才知道为毛下面的代码会出错误，原来是clf没有fit的缘故咯
        #print_nnclf_acc(clf, X_split_validate, Y_split_validate)
        #所以必须进行fit,而且fit函数必须在predict函数中比较真实咯
        #但是我觉得这个是不是有点不公平，毕竟前者cv出结果的时候可能还用到了部分测试集
        #但是怎么可能不公平呢，大家都没有用到验证集的数据吧，有一点不公平的在于这份代码没用best_model
        best_model.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong))
        
        print_nnclf_acc(best_model, X_split_train, Y_split_train)
        
        print_nnclf_acc(best_model, X_split_validate, Y_split_validate)

    #前面只使用cv的方式评价一个较好的模型，但是由于有部分数据未参加训练
    #所以现在在对已经训练好的模型，在进行一次拟合希望能够弥补部分数据的缺少训练
    #用best_model对于拟合模型在进行一次计算咯
    best_model.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 

    print_nnclf_acc(best_model, X_train_scaled, Y_train)
    
    Y_pred = best_model.predict(X_test.values.astype(np.float32))
    #将得到的预测结果写入到文件中去咯
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
    output.to_csv("C:\\Users\\win7\\Desktop\\Titanic_Prediction.csv", index=False)
    print("prediction file has been written.")


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
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=6, trials=trials)
print_best_params_acc(trials)

predict(trials, space_nodes, best_nodes, max_evals=10)