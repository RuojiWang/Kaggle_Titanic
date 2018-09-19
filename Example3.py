#coding=utf-8
#下面两句不放在开头居然就会报错，这也太炫酷了吧
from __future__ import print_function
from __future__ import division
#阿尔法狗就是使用贝叶斯优化的方式自动调参的，朕深感欣慰
#为了理解这个贝叶斯优化，朕也是查阅了很多的资料才看懂一点皮毛
#下面是一些具体的网页学习资源，前两个是贝叶斯优化最后一个是高斯回归过程
#https://www.ctolib.com/topics-122267.html
#https://blog.csdn.net/Snail_Ren/article/details/79005069
#http://www.liulizhe.com/2017/05/09/%E9%AB%98%E6%96%AF%E8%BF%87%E7%A8%8B%EF%BC%88gauss-process%EF%BC%89/
#https://zhuanlan.zhihu.com/p/29779000 这一个里面包含了贝叶斯调参库的一些资料咯
#我还是先调试一下google的这个advisor，如果收费或者调不出来那么还是使用zhusuan或者Hyperopt

"""
#google的这个advisor感觉使用起来非常的奇葩
from advisor_client.model import Study
from advisor_client.model import Trial
from advisor_client.model import TrialMetric
from advisor_client.client import AdvisorClient

client = AdvisorClient()

# Create Study
name = "Study"
study_configuration = {
    "goal":
    "MAXIMIZE",
    "maxTrials":
    5,
    "maxParallelTrials":
    1,
    "params": [{
        "parameterName": "hidden1",
        "type": "INTEGER",
        "minValue": 40,
        "maxValue": 400,
        "scallingType": "LINEAR"
    }]
}
#I find out the reason of this error. Must start server before client. 
#if no server running, the client will meet ConnectionRefusedError.
#I made a low level mistake. Close this issue, please.
#麻痹，使用这个必须先运行服务器然后再运行客户端，感觉很尼玛的麻烦呢，还是算了吧。
#ZhuSuan感觉很吊的样子不愧是清华大学主导研发的，但是主要的问题就是其基于tf很麻烦呢
#之后留意一下这两个吧：jaberg/hyperopt, 比较简单。fmfn/BayesianOptimization， 比较复杂，支持并行调参。

study = client.create_study(name, study_configuration)
print(study)
print(client.list_studies())

trials = client.get_suggestions(study.id, 3)
print(trials)
print(client.list_trials(study.id))
"""

"""
#下面几个实验都是关于hyperopt的实验
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
#ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
#ppn.fit(X_train_std, y_train)

#y_pred = ppn.predict(X_test_std)
#print accuracy_score(y_test, y_pred)

def percept(args):
    global X_train_std,y_train,y_test
    ppn = Perceptron(n_iter=int(args["n_iter"]),eta0=args["eta"]*0.01,random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial
space = {"n_iter":hp.choice("n_iter",range(30,50)),
         "eta":hp.uniform("eta",0.05,0.5)}
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=100)
print(best)
print(percept(best))
"""

"""
#coding:utf-8
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

def loadFile(fileName = "E://zalei//browsetop200Pca.csv"):
    data = pd.read_csv(fileName,header=None)
    data = data.values
    return data

data = loadFile()
label = data[:,-1]
attrs = data[:,:-1]
labels = label.reshape((1,-1))
label = labels.tolist()[0]

minmaxscaler = MinMaxScaler()
attrs = minmaxscaler.fit_transform(attrs)

index = range(0,len(label))
shuffle(index)
trainIndex = index[:int(len(label)*0.7)]
print len(trainIndex)
testIndex = index[int(len(label)*0.7):]
print len(testIndex)
attr_train = attrs[trainIndex,:]
print attr_train.shape
attr_test = attrs[testIndex,:]
print attr_test.shape
label_train = labels[:,trainIndex].tolist()[0]
print len(label_train)
label_test = labels[:,testIndex].tolist()[0]
print len(label_test)
print np.mat(label_train).reshape((-1,1)).shape


def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    print "max_depth:" + str(max_depth)
    print "n_estimator:" + str(n_estimators)
    print "learning_rate:" + str(learning_rate)
    print "subsample:" + str(subsample)
    print "min_child_weight:" + str(min_child_weight)
    global attr_train,label_train

    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm,attr_train,label_train,cv=5,scoring="roc_auc").mean()
    print metric
    return -metric

space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print best
print GBM(best)
"""

"""
#贝叶斯优化的力量真的是完爆调参工程师咯，确实是太炫酷了吧
#而且这个训练速度之快秒出结果超乎我预期，居然能够达到87%88%的准确率
#我现在的担心就是不知道这个框架能够应用到深度学习中，据说可能效果并不佳？
#接下来还有两个问题需要处理：
#（1）搞清楚这两个框架的大致原理以及情况
#（2）比较这两个框架并选择其中一个框架用于以后的机器学习
#（3）将这两个框架用于深度学习并选择其中的一个用于竞赛
#（4）可能学习ZhuSuan以及tensorflow实现竞赛操作吧？
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

from skorch.net import NeuralNetClassifier
import sys
import torch.nn.init
#引入这个东西主要是为了查看MLPClassifier的属性集吧
from sklearn.neural_network import MLPClassifier

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

X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.3, shuffle=True)
    
skf = StratifiedKFold(Y_split_train, n_folds=10, shuffle=True, random_state=None)

def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    
    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    print()
    #global X_train_scaled, Y_train
    global X_split_train, Y_split_train

    #这个应该是设置初值的缘故吧，带入初值进行贝叶斯优化，而不是计算这个初值的结果
    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    #metric = cross_val_score(gbm, X_train_scaled, Y_train, cv=5, scoring="roc_auc").mean()
    #原来交叉验证的值还有这种用法感觉非常的炫酷呢，roc_auc的用法也是比较少见可能因为数据集的因素吧
    metric = cross_val_score(gbm, X_split_train, Y_split_train, cv=5, scoring="roc_auc").mean()
    print(metric)
    return -metric

#这些东西以及解释都出现在fmin.py里面
space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }

#algo是搜索函数，partial之类的是真的没看懂啥意思
#hp.randint之类的东西完全被装饰器修饰的，很难看懂
#max_evals似乎是执行贝叶斯优化的次数的意思咯？
#果然是最大优化次数的意思，只需要对比输出就能够看出来
algo = partial(tpe.suggest,n_startup_jobs=10)
#这个fmin函数就只有前四个参数是必填的，其他的有默认参数不是必填
#我大概看了一下就是这四个参数感觉比较重要其他的无所谓吧。
best = fmin(GBM, space, algo=algo, max_evals=10)

#果然并不需要print(best)只需要best就能够输出结果
print(best)
#这个用法还是有点意思的，因为fmin的return是GBM的参数
print(GBM(best))
"""

"""
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

from skorch.net import NeuralNetClassifier
import sys
import torch.nn.init
#引入这个东西主要是为了查看MLPClassifier的属性集吧
from sklearn.neural_network import MLPClassifier

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

X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.3, shuffle=True)
    
skf = StratifiedKFold(Y_split_train, n_folds=10, shuffle=True, random_state=None)

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization

# Load data set and target values
data, target = make_classification(
    n_samples=1000,
    n_features=45,
    n_informative=12,
    n_redundant=7
)

def svccv(C, gamma):
    val = cross_val_score(
        SVC(C=C, gamma=gamma, random_state=2),
        data, target, 'f1', cv=2
    ).mean()
    return val

def rfccv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFC(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        data, target, 'f1', cv=2
    ).mean()
    return val

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}

    svcBO = BayesianOptimization(svccv,
        {'C': (0.001, 100), 'gamma': (0.0001, 0.1)})
    svcBO.explore({'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})

    rfcBO = BayesianOptimization(
        rfccv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999)}
    )

    svcBO.maximize(n_iter=10, **gp_params)
    print('-' * 53)
    rfcBO.maximize(n_iter=10, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('RFC: %f' % rfcBO.res['max']['max_val'])
"""

"""
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

from skorch.net import NeuralNetClassifier
import sys
import torch.nn.init
#引入这个东西主要是为了查看MLPClassifier的属性集吧
from sklearn.neural_network import MLPClassifier

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

X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.3, shuffle=True)
    
skf = StratifiedKFold(Y_split_train, n_folds=10, shuffle=True, random_state=None)


#Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
#for Mean Absoulte Error objective
#on default features for https://www.kaggle.com/c/allstate-claims-severity

#今天要做的事情：
#（1）了解这个库的原理，并比较和前面库的优劣
#（2）然后准备提交Titanic的结果咯
#（3）然后是提交数字识别结果
#（4）然后是研究一哈特征工程如何做咯
#用了这个库之后似乎这两个贝叶斯优化的库高下立判咯。
#现在正在使用的这个库确实复杂一些，但是感觉吊一些的。
#毕竟这个东西能够实现early stopping感觉就很炫酷
__author__ = "Vladimir Iglovikov"

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm

#我现在的疑问大概就是下面几个：
#（1）就是下面函数每次的初值从哪里来的？ 这个尼玛的调试半天没看懂
#（2）if ___name___里面的params到底是哪里再用？ 直接注销掉可以发现是xgb_evaluate函数在使用，所以封装到该函数内部
#（3）然后如何修改均方差到准确率呢？ 这个准确率其实就是xgb的一个参数，直接参考xgb的评价准则并修改成准确率就完事儿了
#不是上面说的那么简单就可以直接使用准确率咯，这个库实现的内部不知道哪里肯定是用了,mae的直接改xgb的准则就报错
def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    num_rounds = 3000
    random_state = 2016
    
    params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        #'eval_metric': 'merror',
        #'eval_metric': 'error',
        'verbose_eval': True,
        'seed': random_state
    }
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    #cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
    #我勒个去原来这里的early stopping是这样实现的？还有就是这个callbacks函数是自带的吧？
    #我记得skorch也可以实现callbacks吧之前有想过用skroch实现early stopping但是没有具体实现。
    #xgb.callback.early_stop我查了一下代码居然是一个class，回头查一下skorch的callback呢？
    #我最后通过Google搜索查到最新版本的skorch里面的神经网络其实含有这个early-stopping的callback
    xgtrain = xgb.DMatrix(X_train_scaled, Y_train)
    cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mae-mean'].values[-1]


#这里主要在处理读入的数据咯，由于我想测试这个库对于Titanic数据的优化程度，所以并没有用个函数
def prepare_data():
    train = pd.read_csv('../input/train.csv')
    categorical_columns = train.select_dtypes(include=['object']).columns

    for column in tqdm(categorical_columns):
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])

    y = train['loss']

    X = train.drop(['loss', 'id'], 1)
    xgtrain = xgb.DMatrix(X, label=y)

    return xgtrain


if __name__ == '__main__':
    #为了检验这个贝叶斯优化效果如何，我已经将其运行在Titanic的数据集上
    #xgtrain = prepare_data()
    #我记得在这种写法在windows下面才能够执行并行计算吧
    #但是就参数而言，我没看出并行计算的相关参数在哪里？
    #不仅仅是并行计算的参数，这个函数的参数基本就是靠LEGB解释的参数吧？

    #num_rounds = 3000
    #random_state = 2016
    #上面两个函数也是仅仅使用在xgb_evaluate，将其移上去好了
    num_iter = 20
    init_points = 5
    
    #这个params与xgtrain都是用在xgb_evaluate函数内部的
    #我觉得这样的写法不够规范所以我准备将这两个作为函数的参数传入
    #params = {
    #    'eta': 0.1,
    #    'silent': 1,
    #    'eval_metric': 'mae',
    #    'verbose_eval': True,
    #    'seed': random_state
    #}

    #xgtrain原来是用在xgb_evaluate中的，进而使用在xgbBO中的，
    #感觉很奇怪的地方在于：这些部分之间并没有明显的传递数据呢。
    #这个函数第一个参数是待求解最大值的函数，第二参数是字典里面包含参数的最大最小值。
    #其实这里这是定义了一个贝叶斯优化的实例而已，真正的计算完全是在xgbBO.maximize的部分咯
    #这些所谓的计算还包括了xgb_evaluate函数，调用该函数评估应该是maximize中的步骤吧
    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })
    
    #看了这串代码突然感觉自己又忘记了高斯过程和贝叶斯优化的核心思想了
    #https://blog.csdn.net/a769096214/article/details/80920304
    #这里面详细介绍了随机过程、高斯过程的概念以及用核函数的方式拓展高斯过程对于其他过程的表示
    #贝叶斯优化的核心就是利用了核函数的高斯过程对于待优化的函数f(x)进行估计和更新
    #这样的f(x)函数是没有解析表达或者形式未知，故而没办法用跟梯度有关的优化方法
    #从上述的高斯过程可以知道，通过采样可以得到函数f(x)的概率描述，我们可以通过采样更加精确的描述f(x)
    #采样的方式具体的说有两种，一种是开发另一种叫做探索，开发一般指取样目前已知（较大）点周围的点，
    #探索通常是指取样之前未开发区间的点（探索新区间内的点），这两种方式都能够更加精确的描述f(x)一定概率得到其最大值
    #acquisition function就是具体取采样点的函数，他一般会在开发和探索中找到一个平衡点。
    #在具体的数据里面探索是选择方差较大的区域进行取样，而开发是选择更接近均值的区域进行采样咯
    
    #这个函数的文档写的并不清楚，写在上面的参数远没有实际那么多，而且有些有初始值的没写清楚
    #下面这个函数在进行贝叶斯优化，贝叶斯优化的核心就是利用高斯过程不断的探索开发采样点
    #那么上面的xgbBO = BayesianOptimization这个函数在作甚？估计是随机计算采样点吧？
    #上面的函数就是定义了贝叶斯优化类的一个实例而已。。我可能今天工作太久头有点晕了没看懂。。
    #卧槽下面调试的这个代码感觉这个代码结构蛮复杂的到处跳转。。
    xgbBO.maximize(init_points=init_points, n_iter=num_iter)

# Finally, we take a look at the final results.
#type(xgbBO.res)返回的结果果然是dict咯
#print()
#print(xgbBO.res['max'])
#print()
#print(xgbBO.res['all'])
#print()
#print(xgbBO.res)
#print()
print()
print(xgbBO.res['max'])
"""

#现在开始对以上两个贝叶斯优化的库作比较咯
#第一个贝叶斯优化库还可以用，第二个感觉完全没办法用咯
#我真傻比没必要这两个都试一下的吧，这两个实现的是同一个东西
#第二个贝叶斯优化库封装做的一塌糊涂，单步调试非常晦涩难懂，难以大用。
