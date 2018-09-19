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
    max_epochs=10,
    #criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[skorch.callbacks.EarlyStopping(patience=8)]
)

params = {
    'max_epochs':[40, 60, 80, 100, 120],#, 120, 200, 240],
          
    'lr': [0.0001, 0.0002, 0.0005, 0.001],#, 0.002, 0.005, 0.01, 0.02, 0.05],

    'optimizer__weight_decay':[0, 0.001, 0.002, 0.005, 0.01],
              
    'criterion':[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
    
    'batch_size':[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    
    'optimizer__betas':[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], [0.94, 0.999],
        [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]],
        
    'module':[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
              module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
}

X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.01, shuffle=True)

skf = StratifiedKFold(Y_split_train, n_folds=20, shuffle=True, random_state=None)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
#我对这个库的疑问主要体现在下面的这些部分：
#（0）GBM函数是什么时候被调用的
#GBM的函数在fmin.py中的rval.exhaust()被调用执行
#（1）GBM中的参数是如何传进来的，这些值为什么是这样？ 
#单步进入best = fmin(GBM, space, algo=rand.suggest, max_evals=20)
#fmin.py中的domain = base.Domain(fn, space,pass_expr_memo..其space即为参数
#（2）贝叶斯优化中的初始采样点在哪里？
#（3）如何使用搜索到的最优参数设置于模型之中并用于预测
#这个问题的答案参考GBM函数，将参数类似argsDict传入不修改即可。
def GBM(argsDict):

    #我个人感觉这个写法其中有诈呀
    #max_depth = argsDict["max_depth"] + 5
    #n_estimators = argsDict['n_estimators'] * 5 + 50
    #learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    #subsample = argsDict["subsample"] * 0.1 + 0.7
    #min_child_weight = argsDict["min_child_weight"]+1
    
    max_depth = argsDict["max_depth"]
    n_estimators = argsDict['n_estimators']
    learning_rate = argsDict["learning_rate"]
    subsample = argsDict["subsample"]
    min_child_weight = argsDict["min_child_weight"]

    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    print()
    global X_train_scaled, Y_train
    #global X_split_train, Y_split_train

    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    #metric = cross_val_score(gbm, X_train_scaled, Y_train, cv=5, scoring="roc_auc").mean()
    #之前设置的是scoring="roc_auc"我看到87点几的数值还以为是正确率白开心了一下。。
    #cv已经设置到了100啦，而且max_evals已经设置为50了但是上线已经出现了大概就在0.836左右。。
    #我不知道是xgb的训练非常的迅速还是因为cross_val_score与超参搜索中cv的方式不同咯？
    #我觉得应该是相同的吧，都会执行cv所对应的那么多次然后计算出平均值，这个单步调试可以证明
    #另外，从cross_validation中的下面这行代码也可以得到对于上述猜想的的印证
    #scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
    #那么老子有个问题呀，那么xgb之类模型是如何初始化的呢？可能类似lr之类的吧不设置就用默认值咯
    #我目测应该xgboost和lr应该都是如果不设置参数就是用默认值，这样初始化的吧，毕竟才几个参数不像神经网络
    #lr = LogisticRegression()我特地用了用了lr单步调试了其内部的代码lr的初始化（init）确实如前面文字描述
    metric = cross_val_score(gbm, X_train_scaled, Y_train, cv=10, scoring="accuracy").mean()
    print(metric)
    return -metric

space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
    
#hp.randint单步调试时候进入到pull_utils.py中的def validate_label(f)
#进一步单步调试能够发现该函数内部return f(label, *args, **kwargs)
#其中这个f即为hp.randint，args为(10,)，kwargs为空的。这尼玛有点晦涩
#以后建议是用hp.py中的hp.uniform('x', 0, 1)用法代替下面的写法吧。
print(hp.randint("n_estimatirs",10))
#感觉对于这个库的工作过程又有点不清楚了，下午单步调试一下吧。
algo = partial(tpe.suggest, n_startup_jobs=10)
#这个函数果然仅仅只能够返回出最佳的参数并不能够克隆最佳模型呢
#这个函数的最后一个参数就是执行迭代的次数吧，我觉得可以改大一点
#best = fmin(GBM, space, algo=algo, max_evals=50)
#我打算在fmin中加入注释方便代码调试的，加入注释就无法调试删除注释才可以
#fmin函数中的rval.exhaust()输出训练的结果情况咯，可能初始采样点也是这里面实现的
#初始采样点等信息被封装在了fmin函数外部不知如何修改咯，怎么会不知道调？当然是子问题呀。。
#这个函数大致分为两个部分，首先是表示计算空间中某点的值：domain = base.Domain(fn, space,
#然后是对于空间中的点进行搜索和评估：rval = FMinIter(algo, domain, trials, max_evals
#然后调用fmin.py中的exhaust函数内部的self.run(self.max_evals..中的self.serial_evaluate()评估函数
#至于评估函数内的特征空间的点其实是源自于serial_evaluate(self, N=-1)中的spec = base.spec_from_misc(trial['misc'])
#其实所有搜索的点均保存在Trials类中，写为best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)即可
#然后评估函数默认都是只带有params的，怪不得前面的传入训练集的方式这么的畸形，至于trials点的选择就是TPE之类的算法决定的啦
#我感觉自己单步调试费了好大的力气才把这些问题理清楚，外加稍微看了一些文档感觉对于这个库如何使用更加清晰一些了。
best = fmin(GBM, space, algo=algo, max_evals=20)

print(best)
#既然GBM可以GBM(best)可以输出计算结果，那么也能用这个办法
#将最佳参数传递给模型并让模型对于问题进行预测的吧
print(GBM(best))
