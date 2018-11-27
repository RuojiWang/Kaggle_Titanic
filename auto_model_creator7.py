#coding=utf-8
#这个版本应该就是实现两次的超参搜索了吧
#如果真的实现两次超参搜索应该结果会有所提升的吧
#所以我准备在这个版本里面重构代码实现两次超参搜索咯
#然后对于噪声还有相关参数的设置进行了很多的测试，最后发现稳定的参数设置大致是这个样子的
#让我觉得很惊喜的一点就是节点数目的增加将极大的提高最终输出的结果，但是有点担心过拟合
#所以我现在在接下来的版本中准备做一些过拟合相关的测试，还有nodes_list中多个相同节点的准确率咯
import os
import sys
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

sys.path.append("D:\\Workspace\\Titanic")

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import KFold, RandomizedSearchCV

import skorch
from skorch import NeuralNetClassifier

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from tpot import TPOTClassifier

from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.linear_model import LogisticRegression
#下面的这个kfold是实现k折交叉的功能，返回每次的indice，可以设置为shuffle但默认未设
#然后这个StratifiedKFold是返回k折交叉的迭代器，每次通过迭代器返回结果，可以设置为shuffle
#两者的区别在于前者返回indice或者索引列表后者直接返回迭代器，虽然我这一份代码两种方式都有但是让他们并存吧
#from sklearn.model_selection import KFold,StratifiedKFold

warnings.filterwarnings('ignore')
#下面的这个kfold是实现k折交叉的功能，返回每次的indice，可以设置为shuffle但默认未设
#然后这个StratifiedKFold是返回k折交叉的迭代器，每次通过迭代器返回结果，可以设置为shuffle
#两者的区别在于前者返回indice或者索引列表后者直接返回迭代器，虽然我这一份代码两种方式都有但是让他们并存吧
#from sklearn.model_selection import KFold,StratifiedKFold

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

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

def cal_acc(Y_train_pred, Y_train):
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc
    
def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc

def print_nnclf_acc(acc):
    
    print("the accuracy rate of the model on the whole train dataset is:", acc)
  
def print_best_params_acc(trials):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    
    trials_list.sort(key=lambda item: item["result"]["loss"])
    
    print("best parameter is:", trials_list[0])
    print()
    
def exist_files(title):
    
    return os.path.exists(title+"_best_model.pickle")
    
def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params(title):
  
    files = open(str(title+"_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes
    
def save_best_model(best_model, title):
    
    files = open(str(title+"_best_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()
    
def load_best_model(title_and_nodes):
    
    files = open(str(title_and_nodes+"_best_model.pickle"), "rb")
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

def create_module(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list = []
    
    if(hidden_layers==0):
        
        module_list.append(nn.Linear(input_nodes, output_nodes))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
        
    else :
        module_list.append(nn.Linear(input_nodes, hidden_nodes))
        module_list.append(nn.ReLU())
        
        for i in range(0, hidden_layers):
            module_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            module_list.append(nn.ReLU())
             
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
        
    temp_list = []
    for i in range(0, len(module_list)):
        temp_list.append(module_list[i])
        if((i%3==2) and (i!=len(module_list)-2) and (i!=len(module_list)-1)):
            temp_list.append(nn.Dropout(percentage))
            
    model = nn.Sequential()
    for i in range(0, len(temp_list)):
        model.add_module(str(i+1), temp_list[i])
    
    return model

def init_module(clf, weight_mode, bias):
    
    for name, params in clf.named_parameters():
        if name.find("weight") != -1:
            if (weight_mode==1):
                pass
        
            elif (weight_mode==2):
                torch.nn.init.normal_(params)
        
            elif (weight_mode==3):
                torch.nn.init.xavier_normal_(params)
        
            else:
                torch.nn.init.xavier_uniform_(params)
        
        if name.find("bias") != -1:
            if (weight_mode==1):
                pass
        
            elif (weight_mode==2):
                torch.nn.init.constant_(params, bias)
        
            elif (weight_mode==3):
                torch.nn.init.constant_(params, bias)
        
            else:
                torch.nn.init.constant_(params, bias)
        
def noise_augment_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] +=  random.gauss(mean, std)

    return X_noise_train, Y_train

#我有时候再想这个超参是不是有的时候应该重复两次以避免漏掉最佳的超参？？
#或者我换个角度看待这个问题：即便是这个超参是全局最优但是太容易出问题也不行吧
#所以从这个角度出发的话，我觉得用这种方式代表超参其实也是有道理的吧。
#OK，现在新的问题又产生了，现在的问题是：nn_f的函数不支持传递参数
#为了支持两次超参搜索现在有两种方式实现一种方式是写两个类似nn_f的函数，
#还有一种方式是在执行nn_f之前将其训练集的变量指向另外一个训练集就vans了
#我觉得从代码的维护性等方面来说应该后一种方式是更恰当的方式吧，反正也需要重构代码咯
#但是由于这边增加噪声的情况下，所以nn_f并不能够实现通用，看来最后还是要写两份呀
def nn_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("bias", params["bias"])
    print("weight_mode", params["weight_mode"])
    print("patience", params["patience"])
    print("input_nodes", params["input_nodes"])
    print("hidden_layers", params["hidden_layers"])
    print("hidden_nodes", params["hidden_nodes"])
    print("output_nodes", params["output_nodes"])
    print("percentage", params["percentage"])
        
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              module = create_module(params["input_nodes"], params["hidden_layers"], 
                                                      params["hidden_nodes"], params["output_nodes"], params["percentage"]),
                              max_epochs = params["max_epochs"],
                              callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                              device = params["device"],
                              optimizer = params["optimizer"]
                              )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    init_module(clf.module, params["weight_mode"], params["bias"])
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric

def nn_stacking_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("bias", params["bias"])
    print("weight_mode", params["weight_mode"])
    print("patience", params["patience"])
    print("input_nodes", params["input_nodes"])
    print("hidden_layers", params["hidden_layers"])
    print("hidden_nodes", params["hidden_nodes"])
    print("output_nodes", params["output_nodes"])
    print("percentage", params["percentage"])
    
    #这边的columns可以加入所有的选择部分
    #但是先试一下不加和全家之间的区别呢？
    #X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], stacked_train, Y_train, columns=[i for i in range(0, stacked_train.columns.size)])
    #X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], stacked_train, Y_train, columns=[])
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], stacked_train, Y_split_train, columns=[])
   
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              #为了不再重新创建space,space_nodes就用下面的写法吧
                              module = create_module(stacked_train.columns.size, params["hidden_layers"], 
                                                      params["hidden_nodes"], params["output_nodes"], params["percentage"]),
                              max_epochs = params["max_epochs"],
                              callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                              device = params["device"],
                              optimizer = params["optimizer"]
                              )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    init_module(clf.module, params["weight_mode"], params["bias"])
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric
    
def parse_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[0]["misc"]["vals"]["max_epochs"][0]]

    best_nodes["lr"] = space_nodes["lr"][trials_list[0]["misc"]["vals"]["lr"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]]
    best_nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[0]["misc"]["vals"]["weight_mode"][0]]
    best_nodes["bias"] = space_nodes["bias"][trials_list[0]["misc"]["vals"]["bias"][0]]
    best_nodes["patience"] = space_nodes["patience"][trials_list[0]["misc"]["vals"]["patience"][0]]
    best_nodes["device"] = space_nodes["device"][trials_list[0]["misc"]["vals"]["device"][0]]
    best_nodes["optimizer"] = space_nodes["optimizer"][trials_list[0]["misc"]["vals"]["optimizer"][0]]
    
    #新添加的这些元素用于控制模型的结构
    best_nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[0]["misc"]["vals"]["input_nodes"][0]]
    best_nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[0]["misc"]["vals"]["hidden_layers"][0]]
    best_nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[0]["misc"]["vals"]["hidden_nodes"][0]]
    best_nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[0]["misc"]["vals"]["output_nodes"][0]]
    best_nodes["percentage"] = space_nodes["percentage"][trials_list[0]["misc"]["vals"]["percentage"][0]]

    return best_nodes

#我发现了这个程序的一个BUG咯气死我了怪不得没啥好结果
def parse_trials(trials, space_nodes, num):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    #nodes = {}nodes如果在外面那么每次更新之后都是一样的咯
    nodes_list = []
    
    for i in range(0, num):
        nodes = {}
        nodes["title"] = space_nodes["title"][trials_list[i]["misc"]["vals"]["title"][0]]
        nodes["path"] = space_nodes["path"][trials_list[i]["misc"]["vals"]["path"][0]]
        nodes["mean"] = space_nodes["mean"][trials_list[i]["misc"]["vals"]["mean"][0]]
        nodes["std"] = space_nodes["std"][trials_list[i]["misc"]["vals"]["std"][0]]
        nodes["batch_size"] = space_nodes["batch_size"][trials_list[i]["misc"]["vals"]["batch_size"][0]]
        nodes["criterion"] = space_nodes["criterion"][trials_list[i]["misc"]["vals"]["criterion"][0]]
        nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[i]["misc"]["vals"]["max_epochs"][0]]
        nodes["lr"] = space_nodes["lr"][trials_list[i]["misc"]["vals"]["lr"][0]] 
        nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[i]["misc"]["vals"]["optimizer__betas"][0]]
        nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[i]["misc"]["vals"]["optimizer__weight_decay"][0]]
        nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[i]["misc"]["vals"]["weight_mode"][0]]
        nodes["bias"] = space_nodes["bias"][trials_list[i]["misc"]["vals"]["bias"][0]]
        nodes["patience"] = space_nodes["patience"][trials_list[i]["misc"]["vals"]["patience"][0]]
        nodes["device"] = space_nodes["device"][trials_list[i]["misc"]["vals"]["device"][0]]
        nodes["optimizer"] = space_nodes["optimizer"][trials_list[i]["misc"]["vals"]["optimizer"][0]]
        nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[i]["misc"]["vals"]["input_nodes"][0]]
        nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[i]["misc"]["vals"]["hidden_layers"][0]]
        nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[i]["misc"]["vals"]["hidden_nodes"][0]]
        nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[i]["misc"]["vals"]["output_nodes"][0]]
        nodes["percentage"] = space_nodes["percentage"][trials_list[i]["misc"]["vals"]["percentage"][0]]
        
        nodes_list.append(nodes)
    return nodes_list

def nn_model_train(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #由于神经网络模型初始化、dropout等的问题导致网络不够稳定
    #解决这个问题的办法就是多重复计算几次，选择其中靠谱的模型
    best_acc = 0.0
    best_model = 0.0
    for j in range(0, max_evals):
        
        clf = NeuralNetClassifier(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(clf.module, nodes["weight_mode"], nodes["bias"])
        clf.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
            
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)        
    
    return best_model, best_acc

def get_oof(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_acc = []
    valida_acc = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_acc = nn_model_train(nodes, X_split_train, Y_split_train, max_evals)
            
        acc1 = cal_nnclf_acc(best_model, X_split_train, Y_split_train)
        print_nnclf_acc(acc1)
        train_acc.append(acc1)
        acc2 = cal_nnclf_acc(best_model, X_split_valida, Y_split_valida)
        print_nnclf_acc(acc2)
        valida_acc.append(acc2)
        
        oof_train[valida_index] = best_model.predict(X_split_valida.astype(np.float32))
        oof_test_all_fold[:, i] = best_model.predict(X_test_scaled.astype(np.float32))
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0

    #我已经将这份代码的best_nodes["title"]由原来的titanic改为stacked_titanic作为新版本
    if (exist_files(best_nodes["title"])):
        best_model = load_best_model(best_nodes["title"]+"_"+str(len(nodes_list)))
        best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_train.values)
         
    for i in range(0, max_evals):
        
        #这边不是很想用nn_model_train代替下面的函数代码
        #因为这下面的代码还涉及到预测输出的问题不好修改
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module = create_module(stacked_train.columns.size, best_nodes["hidden_layers"], 
                                                         best_nodes["hidden_nodes"], best_nodes["output_nodes"], best_nodes["percentage"]),
                                  max_epochs = best_nodes["max_epochs"],
                                  callbacks = [skorch.callbacks.EarlyStopping(patience=best_nodes["patience"])],
                                  device = best_nodes["device"],
                                  optimizer = best_nodes["optimizer"]
                                  )
        
        init_module(clf.module, best_nodes["weight_mode"], best_nodes["bias"])
        
        clf.fit(stacked_train.values.astype(np.float32), Y_train.values.astype(np.longlong))
        
        metric = cal_nnclf_acc(clf, stacked_train.values, Y_train.values)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        if (flag):
            #这个版本的best_model终于是全局的版本咯，真是开森呢。。
            save_best_model(best_model, best_nodes["title"]+"_"+str(len(nodes_list)))
            Y_pred = best_model.predict(stacked_test.values.astype(np.float32))
            
            """
            #原来在预测的时候下面的代码导致了错误，麻痹的之前搞了好久没弄清楚哦
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
            """
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, best_acc, Y_pred
    
#现在直接利用经验参数值进行搜索咯，这样可以节约计算资源
space = {"title":hp.choice("title", ["stacked_titanic"]),
         "path":hp.choice("path", ["C:/Users/1/Desktop/Titanic_Prediction.csv"]),
         "mean":hp.choice("mean", [0]),
         "std":hp.choice("std", [0.10]),
         "max_epochs":hp.choice("max_epochs",[400]),
         "patience":hp.choice("patience", [4,5,6,7,8,9,10]),
         "lr":hp.choice("lr", [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                               0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                               0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                               0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                               0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                               0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                               0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                               0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                               0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                               0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                               0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                               0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                               0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                               0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                               0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                               0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160]),  
         "optimizer__weight_decay":hp.choice("optimizer__weight_decay",[0.000]),  
         "criterion":hp.choice("criterion", [torch.nn.NLLLoss, torch.nn.CrossEntropyLoss]),

         "batch_size":hp.choice("batch_size", [64, 128, 256, 512, 1024]),
         "optimizer__betas":hp.choice("optimizer__betas",
                                      [[0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                       [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                       [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999]]),
         "input_nodes":hp.choice("input_nodes", [9]),
         "hidden_layers":hp.choice("hidden_layers", [0, 1, 2, 3, 4, 5, 6, 7, 8]), 
         "hidden_nodes":hp.choice("hidden_nodes", [5, 10, 15, 20, 25, 30, 35, 40, 
                                                   45, 50, 55, 60, 65, 70, 75, 80, 
                                                   85, 90, 95, 100, 105, 110, 115]), 
         "output_nodes":hp.choice("output_nodes", [2]),
         "percentage":hp.choice("percentage", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]),
         "weight_mode":hp.choice("weight_mode", [1]),
         "bias":hp.choice("bias", [0]),
         "device":hp.choice("device", ["cpu"]),
         "optimizer":hp.choice("optimizer", [torch.optim.Adam])
         }

space_nodes = {"title":["stacked_titanic"],
               "path":["C:/Users/1/Desktop/Titanic_Prediction.csv"],
               "mean":[0],
               "std":[0.10],
               "max_epochs":[400],
               "patience":[4,5,6,7,8,9,10],
               "lr":[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                     0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                     0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                     0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                     0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                     0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                     0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                     0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                     0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                     0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                     0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                     0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                     0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                     0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                     0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                     0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160],
               "optimizer__weight_decay":[0.000],
               "criterion":[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
               "batch_size":[64, 128, 256, 512, 1024],
               "optimizer__betas":[[0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                   [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                   [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999]],
               "input_nodes":[9],
               "hidden_layers":[0, 1, 2, 3, 4, 5, 6, 7, 8], 
               "hidden_nodes":[5, 10, 15, 20, 25, 30, 35, 40, 
                               45, 50, 55, 60, 65, 70, 75, 80, 
                               85, 90, 95, 100, 105, 110, 115], 
               "output_nodes":[2],
               "percentage":[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
               "weight_mode":[1],
               "bias":[0],
               "device":["cpu"],
               "optimizer":[torch.optim.Adam]
               }

#其实本身不需要best_nodes主要是为了快速测试
#不然每次超参搜索的best_nodes效率太低了吧
best_nodes = {"title":"stacked_titanic",
              "path":"path",
              "mean":0,
              "std":0.1,
              "max_epochs":400,
              "patience":5,
              "lr":0.00010,
              "optimizer__weight_decay":0.005,
              "criterion":torch.nn.CrossEntropyLoss,
              "batch_size":128,
              "optimizer__betas":[0.86, 0.999],
              "input_nodes":9,
              "hidden_layers":3, 
              "hidden_nodes":60, 
              "output_nodes":2,
              "percentage":0.15,
              "weight_mode":1,
              "bias":0.0,
              "device":"cpu",
              "optimizer":torch.optim.Adam
              }

"""
#从现在的结果看来应该是最后输出模型的问题咯
#因为前面的模型输出效果都是挺好的准确率挺高的
#我之前还在担心超参搜索的时候使用了所有的数据集
#我使用所有的数据集只是选择超参但是模型训练
#并没有用所有数据集所以不用担心过拟合的问题咯
#现在这个问题最理想的解决方案可能只有一种了，就进行两次超参搜索咯
#现在需要统一训练集的类型以及将训练集写到函数接口上面不然会出乱子
#训练集写到接口上面倒是很容易，但是里面的数据类型真的是乱七八糟的不好管理
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

#第一次超参搜索搜索每层stacking的最佳结构超参
X_train_f = X_train_scaled
Y_train_f = Y_train
best_params = fmin(nn_f, space, algo=algo, max_evals=10, trials=trials)
print_best_params_acc(trials)

#将获得的结构超参数据进行stacking获得新特征
#best_nodes = parse_nodes(trials, space_nodes)
nodes_list = parse_trials(trials, space_nodes, 3)
#这里best_nodes其实没有保存任何重要信息但是还是保存一下吧
save_inter_params(trials, space_nodes, best_nodes, "titanic")
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 10)

#对获得的新特征进行超参搜索选择结构
stacked_trials = Trials()
#这里进行一次特征缩放说不定效果更好呢
X_train_f = stacked_train
Y_train_f = Y_train
best_stacked_params = fmin(nn_f, space, algo=algo, max_evals=10, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
nn_stacking_predict(best_nodes, stacked_train, Y_train, stacked_test, 5, 10)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#下面的模型居然取得了85.29%的正确率，我真的是看到了希望了，看来stacking才是王道呀
#the best accuracy rate of the model on the whole train dataset is: 0.8529741863075196
#有的地方有.values有的地方又没有这个感觉很凌乱还是都用吧
#其实我早就应该知道的，直接把stacked_train之类的变成df吧
algo = partial(tpe.suggest, n_startup_jobs=10)
#好像这边重复增加超参节点结果居然没有改变耶？5个节点结果差不多的效果
#感觉直接增加重复的次数是能够得到最大的提升的意思咯，我试一下提升比较有限吧
#增加计算次数提升不是很明显，但是增加节点数目提升还是有点明显哦
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes, 
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 10)
stacked_trials = Trials()
#既然最后还是分裂为两个版本所以这些不需要了吧
#X_train_f = stacked_train
#Y_train_f = Y_train
#下面的这个写法不行，因为我是真的可能使用以前的trials，如果修改了就不好了吧
#space["input_nodes"]=hp.choice("input_nodes", [len(nodes_list)])
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
#下面这函数一直报错，花了我很多的时间才知道是之前存储的stacked_titanic_best_model的问题
nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 20)
"""

"""
#之前的数据基本都是试探性的测试，下面的数据才是真实的计算吧
#要是今晚上的大计算效果不理想怎么办呢，感觉那就会很绝望呀
#我总是没有办法在这上面在做一次超参搜索了吧，毕竟感觉只能这样了。
#上面居然在平时只有83%正确率的，stacking以后居然达到了85%的正确率，看到希望了
#刚才运行了一下11节点的60次搜索X2外加30次的结果感觉好像很垃圾的样子呢。
#计算了大概45分钟准确率居然只有83.6%咯，看来新的stacking模型可以探索的超参还很多
#我现在觉得影响这个结果的最大的因素应该是节点的数目吧，因为运行了45分钟那次
#因为我试过增加训练模型的
此外我在想stacked_train是否不应该加噪声？
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=0, trials=trials)
print_best_params_acc(trials)

nodes_list = parse_trials(trials, space_nodes, 10)
save_inter_params(trials, space_nodes, best_nodes, "titanic")
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 10)

stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=600, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 100)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
光是存储best_model是没用的因为stacked_train会变
model = load_best_model("stacked_titanic_11")
cal_nnclf_acc(model, stacked_train, Y_train)
"""

"""
#针对nn_stacking_f噪声测试：
#不加入噪声的时候[10, 5jiedian, 5, 10, 20, 20]的结果 0.8361391694725028 0.8361391694725028 0.8215488215488216
#不加入噪声的时候[10, 5jiedian, 5, 10, 40, 40]的结果 0.8338945005611672
#不加入噪声的时候[10, 5jiedian, 11, 10, 20, 20]的结果 0.8327721661054994 0.8428731762065096 0.8338945005611672
#加入噪声的时候[10, 5jiedian, 5, 10, 20, 20]的结果 0.8338945005611672
#加入噪声的时候[10, 5jiedian, 5, 10, 40, 40]的结果 0.8271604938271605
#加入噪声的时候[10, 5jiedian, 11, 10, 20, 20]的结果 0.8383838383838383 0.8428731762065096 0.8417508417508418
#对比第一列数据就会发现有噪声的时候基本完败，关键问题是节点增加与过拟合的关系咯
#我刚才还在想为什么节点增多提升最为明显，原来是每个节点得到的数据增加了= =所以小心过拟合。。
#当噪声的问题得到了解决之后我就准备尝试是否过拟合，是否同一个节点进行stacking呢？？？
#但是现在先解决稳定性的问题吧，我将stacked_features的第三个参数由10改为了20其他的应该不用吧
#加入噪声的时候[10, 5jiedian, 5, 20, 20, 20]的结果 0.8361391694725028 0.8338945005611672 0.8237934904601572 0.8260381593714927
#不加入噪声的时候[10, 5jiedian, 5, 20, 20, 20]的结果0.8338945005611672 0.8338945005611672 0.8316498316498316 0.8294051627384961 0.8316498316498316这个测试一回合计算600次呀
#等经过计算测试这个节点已经稳定的时候再开始下一步过拟合的测试吧，经过这五次的节点计算现在每次的计算结果已经很稳定了吧
#刚才看到了别人在处理titanic数据的使用采用了one-hot编码其实我也可以使用的呢，现在可以开始测试是否过拟合希望能够有好的结果吧
start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]

stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 20)
stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")

nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 20)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#下面的代码主要是为了测试stacked的数据是否需要加入特征缩放
#我个人理解stacked的数据还是应该加入特征缩放的吧
#不对吧，好像stacked的数据是经过
#X_all_scaled = pd.DataFrame(preprocessing.scale(X_all), columns = X_train.columns)
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, random_state=0)
start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)
train_acc = []
valida_acc = []

所以说这个50的参数设置还有点问题咯，从稳定性的角度描述确实应该是设置为50吧
#下面是第二层使用神经网络的结果，后面的数字表示参与计算的节点数目。
#放入相同节点的话，似乎真的只有两个节点的时候效果最好呢。。
#我有个想法就是以后使用lr作为第二层进行计算呢？
0.8256274768824307 0.8059701492537313 #2
0.8322324966974901 0.7910447761194029 #5
0.845442536327609 0.7835820895522388 #11  

0.8282694848084544 0.8134328358208955 #2 
0.8322324966974901 0.7985074626865671 #5
0.8348745046235139 0.7910447761194029 #11

0.821664464993395 0.8059701492537313 #1
0.8282694848084544 0.8059701492537313 #2  
0.8441215323645971 0.8134328358208955 #5   
0.8388375165125496 0.8134328358208955 #11 

0.8243064729194187 0.7835820895522388 #1
0.8229854689564069 0.8134328358208955 #2   
0.8401585204755614 0.7985074626865671 #5   
0.8256274768824307 0.7985074626865671 #11 

0.8295904887714664 0.7985074626865671 #1
0.8243064729194187 0.7910447761194029 #2 
0.8269484808454426 0.7985074626865671 #5 
0.8639365918097754 0.835820895522388 #11

0.8282694848084544 0.8059701492537313 #1
0.8295904887714664 0.8059701492537313 #2
0.8401585204755614 0.8134328358208955 #5
0.8322324966974901 0.8059701492537313 #11

0.8243064729194187 0.7910447761194029 #1
0.8269484808454426 0.8134328358208955 #2
0.8441215323645971 0.7985074626865671 #5
0.8546895640686922 0.7985074626865671 #11

0.8243064729194187 0.7910447761194029 #1
0.8322324966974901 0.7910447761194029 #2
0.8269484808454426 0.8059701492537313 #5
0.8322324966974901 0.8059701492537313 #11
#这部分是一个节点的结果
nodes_list = [best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
#save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
test_acc = cal_acc(Y_train_pred, Y_split_test)
train_acc.append(best_acc)
valida_acc.append(test_acc)

#这部分是两个节点的结果
nodes_list = [best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
#save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
test_acc = cal_acc(Y_train_pred, Y_split_test)
train_acc.append(best_acc)
valida_acc.append(test_acc)

#这部分五个节点的结果
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
#save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
test_acc = cal_acc(Y_train_pred, Y_split_test)
train_acc.append(best_acc)
valida_acc.append(test_acc)

#这部分是十一个节点的结果
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
stacked_trials = Trials()
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
#save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
test_acc = cal_acc(Y_train_pred, Y_split_test)
train_acc.append(best_acc)
valida_acc.append(test_acc)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

#输出的结果大致是这个样子的呢
for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])
"""

"""
#第二层使用逻辑回归是真的更好一些的呢。。
0.8295904887714664 0.7910447761194029 #1
0.8295904887714664 0.7985074626865671

0.8282694848084544 0.8059701492537313 #2
0.8256274768824307 0.8059701492537313

0.8309114927344782 0.8134328358208955 #4
0.8256274768824307 0.8059701492537313

0.8322324966974901 0.8059701492537313 #5
0.8295904887714664 0.7985074626865671

0.8361955085865258 0.8059701492537313 #7
0.8256274768824307 0.7985074626865671

0.845442536327609 0.8059701492537313 #9
0.8243064729194187 0.8059701492537313

0.8256274768824307 0.8208955223880597 #1
0.8256274768824307 0.8208955223880597

0.8295904887714664 0.8059701492537313 #2
0.8295904887714664 0.8059701492537313

0.8137384412153237 0.7686567164179104 #4
0.8124174372523117 0.7761194029850746

0.8309114927344782 0.8059701492537313 #5
0.8177014531043593 0.8059701492537313

0.8322324966974901 0.8134328358208955 #7
0.8229854689564069 0.8283582089552238

0.8361955085865258 0.7985074626865671 #9
0.8229854689564069 0.7985074626865671

0.821664464993395 0.7835820895522388 #1
0.821664464993395 0.7835820895522388

0.8190224570673712 0.7985074626865671 #2
0.8190224570673712 0.7985074626865671

0.8348745046235139 0.7985074626865671 #4
0.8256274768824307 0.8134328358208955

0.8256274768824307 0.7985074626865671 #5
0.8203434610303831 0.8059701492537313

0.8335535006605019 0.8059701492537313 #7
0.8295904887714664 0.8134328358208955

0.8282694848084544 0.7910447761194029 #9
0.821664464993395 0.8059701492537313

0.8177014531043593 0.8059701492537313 #1
0.8177014531043593 0.8059701492537313

0.8256274768824307 0.8059701492537313 #2
0.8243064729194187 0.7985074626865671

0.8348745046235139 0.8208955223880597 #4
0.8203434610303831 0.8283582089552238

0.8322324966974901 0.7910447761194029 #5
0.8243064729194187 0.7985074626865671

0.8295904887714664 0.7910447761194029 #7
0.821664464993395 0.8059701492537313

0.8295904887714664 0.8059701492537313 #9
0.8256274768824307 0.7835820895522388

start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)
train_acc = []
valida_acc = []

for i in range(0, 4):
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, random_state=0)

    #这部分是一个节点的结果
    nodes_list = [best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #这部分是二个节点的结果
    nodes_list = [best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #这部分是四个节点的结果
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #这部分是五个节点的结果
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,  best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #这部分是七个节点的结果
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,  best_nodes, best_nodes,  best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #这部分是九个节点的结果
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,  
                  best_nodes, best_nodes,  best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    print_best_params_acc(stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    #save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 50)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是放入逻辑回归的计算结果咯
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

#输出的结果大致是这个样子的呢
for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])
"""

"""
#那么下面就用之前计算过的超参来进行最佳模型的生成吧
#最迟在周日肯定要提交这个比赛的结果了吧。
#下面使用这个超参搜索得到的titanic_intermediate_parameters_2018-10-1104516.pickle
#或许我现在需要在真实的超参搜索数据上面试一下不同节点最终得到的结果的差异到底如何
#今天很开心的就是找到了量化交易相关的成都招聘了，我总觉得这个才是我真心喜欢的事业呢。。
#那么接下来的事情就是准备调研一下国内的量化交易相关领域的现状如何咯？
#这份代码主要是考虑到要是不是由于相同节点造成的对于性能的影响而建立的测试部分咯
#我现在想要分析的问题大致有两个：节点的数目，以及第二层是否使用逻辑回归
#就这三轮的数据总体而言表现出来的趋势大致是：
#随着节点数目的增多，第一种结构中，训练集的准确率总体在增长，测试集的准确率总体在下降
#随着节点数目的增多，第二种结构中，训练集的准确率总体在增长，测试集的准确率总体在增长
#那么到现在为止，我真的觉得机器学习的很多东西其实就是玄学问题咯
#但是至少有一点是肯定的，就是加入了stacking之后模型确实是有所提升的，这个数据集就选择9个节点吧
0.8177014531043593 0.8208955223880597 #1
0.8177014531043593 0.8134328358208955

0.8190224570673712 0.8059701492537313 #2
0.808454425363276 0.7985074626865671

0.8269484808454426 0.8059701492537313 #3
0.8150594451783355 0.7835820895522388

0.8243064729194187 0.8059701492537313 #4
0.8150594451783355 0.7985074626865671

0.8282694848084544 0.7910447761194029 #5
0.821664464993395 0.8059701492537313

0.8335535006605019 0.7835820895522388 #7
0.8256274768824307 0.8134328358208955

0.8388375165125496 0.7835820895522388 #9
0.8229854689564069 0.8208955223880597

0.8480845442536328 0.7910447761194029 #11
0.821664464993395 0.8208955223880597

0.8467635402906208 0.7910447761194029 #13
0.8269484808454426 0.8283582089552238

0.821664464993395 0.7910447761194029 #1
0.821664464993395 0.7910447761194029

0.8243064729194187 0.8134328358208955 #2
0.8243064729194187 0.7985074626865671

0.8229854689564069 0.7985074626865671 #3
0.8203434610303831 0.8059701492537313

0.8190224570673712 0.7910447761194029 #4
0.8110964332892999 0.8059701492537313

0.8322324966974901 0.8134328358208955 #5
0.8282694848084544 0.8059701492537313

0.8348745046235139 0.7910447761194029 #7
0.8282694848084544 0.8059701492537313

0.8375165125495376 0.7910447761194029 #9
0.8295904887714664 0.8059701492537313

0.8388375165125496 0.8059701492537313 #11
0.8269484808454426 0.7985074626865671

0.8348745046235139 0.7835820895522388 #13
0.8243064729194187 0.8134328358208955

0.8229854689564069 0.8134328358208955 #1
0.8229854689564069 0.8134328358208955

0.8282694848084544 0.8134328358208955 #2
0.8282694848084544 0.8134328358208955

0.8256274768824307 0.7985074626865671 #3
0.8203434610303831 0.8059701492537313

0.8322324966974901 0.7910447761194029 #4
0.8190224570673712 0.7985074626865671

0.8256274768824307 0.8283582089552238 #5
0.8190224570673712 0.8059701492537313

0.8375165125495376 0.7910447761194029 #7
0.8190224570673712 0.7910447761194029

0.8322324966974901 0.7910447761194029 #9
0.8243064729194187 0.8059701492537313

0.8282694848084544 0.8059701492537313 #11
0.8335535006605019 0.7985074626865671

0.8388375165125496 0.7910447761194029 #13
0.821664464993395 0.7835820895522388

files = open("titanic_intermediate_parameters_2018-11-9172110.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
train_acc = []
valida_acc = []

start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, random_state=0)

for i in range(0, 3):
    #下面是一个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 1)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #下面是二个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 2)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #下面是三个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 3)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #下面是四个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 3)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #下面是五个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 5)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    #下面是七个节点的结果咯
    nodes_list = parse_trials(trials, space_nodes, 7)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    nodes_list = parse_trials(trials, space_nodes, 9)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    nodes_list = parse_trials(trials, space_nodes, 11)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    nodes_list = parse_trials(trials, space_nodes, 13)
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 25)
    stacked_trials = Trials()
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_train_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 40)
    test_acc = cal_acc(Y_train_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
#然后这里是输出计算的结果咯
for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])
"""

"""
#其实现在所有的步骤能够通过StackingCVClassifier实现的吧
#只不过当时没有考虑到stacking的过拟合风险罢了吧
#下面的这个实验可以很好的测试不同节点和meta_classifier对于结果的影响
#不行哦，这样子的话所有模型都学习过所有的数据咯，这样子就很容易过拟合滴。。
clf1 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
clf2 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
clf3 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
clf4 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
clf5 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
clf6 = nn_model_train(best_nodes, X_train_scaled, Y_train, max_evals=10)
lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, n_folds=3, verbose=3)
sclf.fit(X_train_scaled, Y_train)
"""

"""
#但是我们可以尝试一下到底是使用逻辑回归作为meta_class比较好还是使用knn算法吧
#但是我发现如果不进行设置的话两种方式得到的结果基本上是差不多的
#今天最大的收获可能真的是关于自己的副业的收获了，感觉遇到了蛮多神器事情的呢
#0.8269484808454426 0.8059701492537313 #一个节点
#0.8269484808454426 0.8059701492537313

#0.821664464993395 0.8134328358208955 #两个节点
#0.821664464993395 0.8208955223880597

#0.8295904887714664 0.8059701492537313 #三个节点
#0.8309114927344782 0.7985074626865671

#0.8243064729194187 0.8059701492537313 #一个节点
#0.8243064729194187 0.8059701492537313

#0.8322324966974901 0.7985074626865671 #两个节点
#0.8322324966974901 0.8059701492537313

#0.8282694848084544 0.7985074626865671 #三个节点
#0.8335535006605019 0.8059701492537313

#0.8269484808454426 0.8208955223880597 #一个节点
#0.8269484808454426 0.8208955223880597

#0.8335535006605019 0.8134328358208955 #两个节点
#0.8335535006605019 0.8134328358208955

#0.8295904887714664 0.8134328358208955 #三个节点
#0.8322324966974901 0.8059701492537313

#0.8243064729194187 0.8059701492537313 #九个节点
#0.8322324966974901 0.8059701492537313

#0.8282694848084544 0.8059701492537313 #九个节点
#0.8295904887714664 0.7985074626865671

#0.8335535006605019 0.7985074626865671 #九个节点
#0.8335535006605019 0.7985074626865671

train_acc = []
valida_acc = []

start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)

for i in range(0, 2):
    #上一个实验居然没有把split的部分放入到循环内，但是好像目前为止没发现啥大的问题吧。
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, random_state=0)
    
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
                  best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    knn = KNeighborsClassifier()
    knn.fit(stacked_train, Y_split_train)
    best_acc = knn.score(stacked_train, Y_split_train)
    knn_pred = knn.predict(stacked_test)
    test_acc = cal_acc(knn_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])
"""

"""
#昨日的大计算结果非常的。。不理想我比较好奇到底怎么才能够提高最后的准确率呢
#我现在的思路主要是想借鉴TPOT的相关经验完成对于这个数据的优化吧
#现在的思路：
#（1）先尝试对于第二层进行超参搜索呢。已经实现过类似的代码了弟弟。
#（2）然后尝试TPOT的相关方法能够进一步优化？
#（3）然后处理一下感情相关的内容
#（4）然后处理一下副业相关的内容
#（5）然后处理一下姑妈的创业问题
#（6）然后处理一下淘宝的问题
#下面的这个数据就是11-12日7000次搜索的结果咯
files = open("titanic_intermediate_parameters_2018-11-13060058.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()

best_nodes = parse_nodes(trials, space_nodes)

train_acc = []
valida_acc = []
time_cost = []

algo = partial(tpe.suggest, n_startup_jobs=10)
#从下面的实验数据而言我觉得总有些东西可以总结出来：
#（0）先看一下是不是有bug，我怎么觉得现在的结果很奇怪呢？并实现数据集样本类型均匀划分
#（1）九个节点的总体而言似乎效果最好
#（2）用九个最佳节点比用前九个节点效果好
#（3）stacking第二层不应该增加噪声的
#（4）第二层使用超参搜索的效果应该较好比不使用的好
#（5）stacking第二层用stacking或者tpot泛化效果没有逻辑回归好。。
#（6）我现在怀疑是不是第二层超参搜索越久越容易过拟合？？一会儿做点实验验证一下。。
#（7）然后我现在觉得真的有必要在第二层lr增加超参搜索咯。。
0.8256274768824307
0.7985074626865671
0.8480845442536328
0.7985074626865671

0.8626155878467635
0.8059701492537313

0.857331571994716
0.8656716417910447

0.821664464993395
0.8283582089552238
0.8401585204755614
0.8134328358208955

0.8348745046235139
0.8059701492537313

0.8401585204755614
0.7910447761194029

0.8229854689564069
0.8134328358208955
0.8414795244385733
0.8283582089552238

0.8507265521796565
0.8059701492537313

0.8295904887714664
0.7985074626865671

0:20:19.970833
0:28:16.862013
3:14:11.963033

0:34:06.591781
0:21:54.658663
1:15:40.699845

0:19:55.276038
0:29:27.091579
3:21:25.778325
for i in range(0, 3):
    
    start_time = datetime.datetime.now()
    #上一个实验居然没有把split的部分放入到循环内，但是好像目前为止没发现啥大的问题吧。
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, random_state=0)
    
    #下面是使用stacking的部分，使用九个best_nodes的那种，分为使用lr和knn的两部分计算
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
                  best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 50)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面的部分才是knn相关计算
    knn = KNeighborsClassifier()
    knn.fit(stacked_train, Y_split_train)
    best_acc = knn.score(stacked_train, Y_split_train)
    knn_pred = knn.predict(stacked_test)
    test_acc = cal_acc(knn_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    start_time = datetime.datetime.now()
    stacked_trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=10)
    #留意这个nn_stacking_f内部的参数哈
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=220, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 100)
    best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_split_train.values)
    test_acc = cal_acc(Y_pred, Y_split_test.values)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)

    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    start_time = datetime.datetime.now()
    #下面是使用stacking然而第二层使用tpot进行计算
    tpot = TPOTClassifier(generations=130, population_size=130, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])

for i in range(0, len(time_cost)):
    print(time_cost[i])
"""

"""
#现在看一下如何实现均匀划分版本的实验咯
#我觉得以后建议使用均匀划分的版本吧，不然泛化性能难以衡量
#虽然比一定每次的比赛都是均匀划分的数据，但是更有意义一些吧
#顺便看一下代码是否存在BUG，虽然我自己内心觉得应该没问题的吧
#下面的计算结果似乎除了一点问题，偶尔会出现只有40%正确率的情况
#但是我走查了几遍应该不是代码的问题，否则会大规模的每次都出现问题
#所以好像神经网络从未出现过类似的问题吧？这也和我几十次计算神经网络选择其中一次有关吧。
#我仔细看了一下这么多的历史数据，确实是tpot偶尔会给出非常奇怪的数据
#所以说tpot好像神奇不在了，在我stacking出来的特征上并不能够给我更好的结果咯？
#神经网络超参搜索的效果也不明显是不是因为我的节点数目过多了呢？
#感觉现在可以做的事情就是节点数目再次确定以及lr的超参搜索部分了吧啊
0.8322324966974901
0.8059701492537313
0.8467635402906208
0.7985074626865671

0.8546895640686922 #tpot900次的计算结果
0.8059701492537313

0.3844121532364597 #tpot1600次的计算结果
0.8059701492537313

0.8494055482166446 #tpot2500次的计算结果
0.6194029850746269

0.8494055482166446 #tpot3600次的计算结果
0.6194029850746269

0.8256274768824307
0.7910447761194029
0.8467635402906208
0.7686567164179104

0.8388375165125496 #tpot900次的计算结果
0.7910447761194029

0.8705416116248349 #tpot1600次的计算结果
0.7835820895522388

0.869220607661823  #tpot2500次的计算结果
0.7910447761194029

0.8480845442536328 #tpot3600次的计算结果
0.7835820895522388

0.8243064729194187
0.835820895522388
0.40290620871862615
0.417910447761194

0.8282694848084544 #tpot900次的计算结果
0.835820895522388

0.8599735799207398 #tpot1600次的计算结果
0.835820895522388

0.8586525759577279 #tpot2500次的计算结果
0.8208955223880597

0.8639365918097754 #tpot3600次的计算结果
0.8283582089552238

0.8388375165125496
0.7164179104477612
0.8520475561426685
0.746268656716418

0.857331571994716  #tpot900次的计算结果
0.7089552238805971

0.8665785997357992 #tpot1600次的计算结果
0.7313432835820896

0.8560105680317041 #tpot2500次的计算结果
0.7611940298507462

0.8507265521796565 #tpot3600次的计算结果
0.7164179104477612

0:08:59.675000
0:13:55.633000
0:15:28.455000
0:08:43.518000
0:13:09.351000
0:12:58.694000
0:14:32.505000
0:17:03.329000
0:24:33.828000
2:08:25.978000
0:10:19.448000
0:09:09.583000
0:11:05.819000
0:16:15.389000
0:29:49.463000
0:08:53.075000
0:03:10.867000
0:03:37.609000
0:27:48.822000
0:09:57.731000
files = open("titanic_intermediate_parameters_2018-11-13060058.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()

best_nodes = parse_nodes(trials, space_nodes)

train_acc = []
valida_acc = []
time_cost = []

algo = partial(tpe.suggest, n_startup_jobs=10)
 
for i in range(0, 4):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, stratify=Y_train)
    
    start_time = datetime.datetime.now()
    #下面是使用stacking的部分，使用九个best_nodes的那种，分为使用lr和knn的两部分计算
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
                  best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 30)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面的部分才是knn相关计算
    knn = KNeighborsClassifier()
    knn.fit(stacked_train, Y_split_train)
    best_acc = knn.score(stacked_train, Y_split_train)
    knn_pred = knn.predict(stacked_test)
    test_acc = cal_acc(knn_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))    
    
    #下面是stacking第二层使用tpot进行计算900次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=30, population_size=30, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    #下面是stacking第二层使用tpot进行计算1600次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=40, population_size=40, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

    #下面是stacking第二层使用tpot进行计算2500次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=50, population_size=50, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

    #下面是stacking第二层使用tpot进行计算3600次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=60, population_size=60, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])

for i in range(0, len(time_cost)):
    print(time_cost[i])
"""    

"""
#这个实验主要验证超参搜索的次数和泛化性能的情况
#具体一点说，我感觉stacking第二层搜索越久泛化性能越差
0.8229854689564069
0.8059701492537313
0.8282694848084544
0.8059701492537313

0.8361955085865258
0.8059701492537313

0.8507265521796565
0.8059701492537313

0.845442536327609
0.8059701492537313

0.8348745046235139
0.7910447761194029

0.8348745046235139
0.7910447761194029

0.8256274768824307
0.7985074626865671

0.8560105680317041
0.7985074626865671

0.8269484808454426
0.7985074626865671

0.8256274768824307
0.8059701492537313


0.8137384412153237
0.8059701492537313
0.8256274768824307
0.8059701492537313

0.8282694848084544
0.7835820895522388

0.8269484808454426
0.7985074626865671

0.8229854689564069
0.7910447761194029

0.8295904887714664
0.7910447761194029

0.8229854689564069
0.8059701492537313

0.8203434610303831
0.7985074626865671

0.8177014531043593
0.7238805970149254

0.8163804491413474
0.8059701492537313

0.8203434610303831
0.7985074626865671

#第一轮计算
0:36:54.450000
0:02:26.068000
0:14:01.811000
0:49:13.769000
6:55:56.201000
1 day, 4:25:31.904000 计算两个10000次的神经网络所花费的时间，以上除了第一个都是神经网络
0:03:29.269000
0:24:05.258000
0:56:05.257000
8:02:11.403000
#第二轮计算
1:02:12.374000
0:04:00.646000
0:11:27.458000
0:44:46.505000
5:02:36.069000
1 day, 13:23:58.172521 计算两个10000次的神经网络所花费的时间，以上除了第一个都是神经网络
0:03:46.392201
0:36:03.225214
1:19:44.659831
9:03:25.993008
#需要说明一下的是我把之前的代码误删了，之前的代码类似这个结构，只不过每次计算的参数有出入
#而且神经网络的计算次数和tpot的计算次数也有一些出入，但是好在从计算时间等信息可以回溯结果
#所以这个实验的结果是第二层计算的时间和泛化性能之间几乎没有任何关系。。看来最终还是要寄出第二层lr超参搜索的神器咯？
files = open("titanic_intermediate_parameters_2018-11-13060058.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()

best_nodes = parse_nodes(trials, space_nodes)

train_acc = []
valida_acc = []
time_cost = []

algo = partial(tpe.suggest, n_startup_jobs=10)

for i in range(0, 2):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, stratify=Y_train)
    
    start_time = datetime.datetime.now()
    #下面是使用stacking的部分，使用九个best_nodes的那种，分为使用lr和knn的两部分计算
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
                  best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 30)
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面的部分才是knn相关计算
    knn = KNeighborsClassifier()
    knn.fit(stacked_train, Y_split_train)
    best_acc = knn.score(stacked_train, Y_split_train)
    knn_pred = knn.predict(stacked_test)
    test_acc = cal_acc(knn_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    #下面是stacking第二层的神经网络超参搜索30次
    start_time = datetime.datetime.now()
    stacked_trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=10)
    #留意这个nn_stacking_f内部的参数哈
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=30, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 30)
    best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_split_train.values)
    test_acc = cal_acc(Y_pred, Y_split_test.values)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    #下面是stacking第二层的神经网络超参搜索120次
    start_time = datetime.datetime.now()
    stacked_trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=10)
    #留意这个nn_stacking_f内部的参数哈
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=120, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 120)
    best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_split_train.values)
    test_acc = cal_acc(Y_pred, Y_split_test.values)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

    #下面是stacking第二层的神经网络超参搜索500次
    start_time = datetime.datetime.now()
    stacked_trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=10)
    #留意这个nn_stacking_f内部的参数哈
    best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=500, trials=stacked_trials)
    best_nodes = parse_nodes(stacked_trials, space_nodes)
    best_model, best_acc, Y_pred = nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_split_train, stacked_test, 500)
    best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_split_train.values)
    test_acc = cal_acc(Y_pred, Y_split_test.values)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
        
    #下面是stacking第二层使用tpot进行计算900次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=30, population_size=30, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))

    #下面是stacking第二层使用tpot进行计算3600次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=60, population_size=60, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    #下面是stacking第二层使用tpot进行计算10000次
    start_time = datetime.datetime.now()
    tpot = TPOTClassifier(generations=100, population_size=100, verbosity = 2)
    tpot.fit(stacked_train, Y_split_train)
    best_acc = tpot.score(stacked_train, Y_split_train)
    tpot_pred = tpot.predict(stacked_test)
    test_acc = cal_acc(tpot_pred, Y_split_test)  
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
        
for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])

for i in range(0, len(time_cost)):
    print(time_cost[i])
"""

"""
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, stratify=Y_train)
start_time = datetime.datetime.now()
clf = LogisticRegression(class_weight={0:549/(549+342),1:342/(549+342)})
param_dist = {"penalty": ["l1", "l2"],
              "C": np.linspace(0.001, 100000, 1000),
              "fit_intercept": [True, False],
              #"solver": ["newton-cg", "lbfgs", "liblinear", "sag"]
             }
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=200)
random_search.fit(X_split_train, Y_split_train)
best_acc = random_search.best_estimator_.score(X_split_train, Y_split_train)
print(best_acc)
test_acc = random_search.best_estimator_.score(X_split_test, Y_split_test)
print(test_acc)
"""

#接下来尝试一下4000的情况，分层划分数据集
#然后尝试一下lr超参的提高程度如何，分层划分数据集
#顺便将就测试一下节点数目对于结果最后的影响呢？
files = open("titanic_intermediate_parameters_2018-11-13060058.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()

best_nodes = parse_nodes(trials, space_nodes)

train_acc = []
valida_acc = []
time_cost = []

algo = partial(tpe.suggest, n_startup_jobs=10)
 
for i in range(0, 3):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, stratify=Y_train)
    
    start_time = datetime.datetime.now()
    #下面是使用stacking的部分，使用九个best_nodes的那种，分为使用lr和knn的两部分计算
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
                  best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    stacked_train, stacked_test = stacked_features(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 30)
    #下面是逻辑回归不进行任何超参搜索
    lr = LogisticRegression()
    lr.fit(stacked_train, Y_split_train)
    best_acc = lr.score(stacked_train, Y_split_train)
    lr_pred = lr.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    #下面是knn不进行任何的超参搜索
    knn = KNeighborsClassifier()
    knn.fit(stacked_train, Y_split_train)
    best_acc = knn.score(stacked_train, Y_split_train)
    knn_pred = knn.predict(stacked_test)
    test_acc = cal_acc(knn_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))    
    
    #下面是针对逻辑回归进行超参搜索的结果
    start_time = datetime.datetime.now()
    clf = LogisticRegression()
    param_dist = {"penalty": ["l1", "l2"],
                  "C": np.linspace(0.001, 100000, 10000),
                  "fit_intercept": [True, False],
                  #"solver": ["newton-cg", "lbfgs", "liblinear", "sag"]
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=2000)
    random_search.fit(stacked_train, Y_split_train)
    best_acc = random_search.best_estimator_.score(stacked_train, Y_split_train)
    lr_pred = random_search.best_estimator_.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
    #下面是针对逻辑回归进行超参搜索的结果，额外设置了class_weight参数
    start_time = datetime.datetime.now()
    clf = LogisticRegression(class_weight={0:549/(549+342),1:342/(549+342)})
    param_dist = {"penalty": ["l1", "l2"],
                  "C": np.linspace(0.001, 1000000, 10000),
                  "fit_intercept": [True, False],
                  #"solver": ["newton-cg", "lbfgs", "liblinear", "sag"]
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=2000)
    random_search.fit(stacked_train, Y_split_train)
    best_acc = random_search.best_estimator_.score(stacked_train, Y_split_train)
    lr_pred = random_search.best_estimator_.predict(stacked_test)
    test_acc = cal_acc(lr_pred, Y_split_test)
    train_acc.append(best_acc)
    valida_acc.append(test_acc)
    end_time = datetime.datetime.now()
    time_cost.append((end_time - start_time))
    
for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])

for i in range(0, len(time_cost)):
    print(time_cost[i])
