#coding=utf-8
#之前已经了解了很多关于stacking、GAN的原理以及构思了方案
#现在这个版本准备实现一个基于超参搜索的神经网络的stacking
#一旦这个版本的代码完成了，剩下的事情就是选择超参让机器自动化计算咯
#我还想再多做点什么，能够让我更加敏捷的实现代码和模型的更新呢？
#所以我又回头看了一下下面的这个博客，我发现我现在的很多方法领先于他
#https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/
#但是我发现了一个问题呀，不同模型之间的做好stacking的基础就是模型有一定差异
#下面两个博客对于集成学习进行了相关的介绍：
#https://blog.csdn.net/a358463121/article/details/53054686
#https://www.sciencedirect.com/science/article/pii/S000437020200190X
#csdn上面的内容使我发现了黑科技。。多人组队每人训练一个模型使用投票的法则。。
#还尼玛可以从提交的历史文件中通过投票的办法选择最佳的模型呢。。
#我现在准备做的事情就是基于目前的神经网络超参搜索实现一次stacking
#如果不理想那么再尝试使用皮尔逊相关系数去解决这个问题咯
import os
import sys
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
from astropy.modeling.tests.test_models import create_model
from networkx.readwrite.json_graph.node_link import node_link_data
sys.path.append("D:\\Workspace\\Titanic")

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

import skorch
from skorch import NeuralNetClassifier

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier

warnings.filterwarnings('ignore')

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

def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
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
    
def load_best_model(title):
    
    files = open(str(title+"_best_model.pickle"), "rb")
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

"""
#运行一下下面的程序看看输出就知道程序到底是咋回事儿了
#我勒个去，刚才一直没搞懂是啥情况呢。
def init_module(clf, weight_mode, bias):
    
    for name, params in clf.named_parameters():
        print(name)
        print(params.size())
"""
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

def parse_trials(trials, space_nodes, num):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    nodes = {}
    nodes_list = []
    
    for i in range(0, num):
        nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
        nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
        nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
        nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
        nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
        nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
        nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[0]["misc"]["vals"]["max_epochs"][0]]
        nodes["lr"] = space_nodes["lr"][trials_list[0]["misc"]["vals"]["lr"][0]] 
        nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
        nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]]
        nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[0]["misc"]["vals"]["weight_mode"][0]]
        nodes["bias"] = space_nodes["bias"][trials_list[0]["misc"]["vals"]["bias"][0]]
        nodes["patience"] = space_nodes["patience"][trials_list[0]["misc"]["vals"]["patience"][0]]
        nodes["device"] = space_nodes["device"][trials_list[0]["misc"]["vals"]["device"][0]]
        nodes["optimizer"] = space_nodes["optimizer"][trials_list[0]["misc"]["vals"]["optimizer"][0]]
        nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[0]["misc"]["vals"]["input_nodes"][0]]
        nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[0]["misc"]["vals"]["hidden_layers"][0]]
        nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[0]["misc"]["vals"]["hidden_nodes"][0]]
        nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[0]["misc"]["vals"]["output_nodes"][0]]
        nodes["percentage"] = space_nodes["percentage"][trials_list[0]["misc"]["vals"]["percentage"][0]]
        
        nodes_list.append(nodes)
    return nodes_list

def predict(best_nodes, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0

    if (exist_files(best_nodes["title"])):
        best_model = load_best_model(best_nodes["title"])
        best_acc = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
         
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module = create_module(best_nodes["input_nodes"], best_nodes["hidden_layers"], 
                                                          best_nodes["hidden_nodes"], best_nodes["output_nodes"], best_nodes["percentage"]),
                                  max_epochs = best_nodes["max_epochs"],
                                  callbacks = [skorch.callbacks.EarlyStopping(patience=best_nodes["patience"])],
                                  device = best_nodes["device"],
                                  optimizer = best_nodes["optimizer"]
                                  )
        
        init_module(clf.module, best_nodes["weight_mode"], best_nodes["bias"])
        
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        if (flag):
            #这个版本的best_model终于是全局的版本咯，真是开森呢。。
            save_best_model(best_model, best_nodes["title"])
            Y_pred = best_model.predict(X_test_scaled.values.astype(np.float32))
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    
#我觉得实现这个细节太多了吧，比如说是如何控制模型的差异？
#还有其实我一直比较担心同样的神经网络的模型这样是否有效？
#可能接下来的工作就是寻找周志华的论文然后阅读了吧，毕竟GASEN
#是否使用全训练集咯？感觉这是一个很重要的问题
#但是模型无论如何还是必须要放在list里面的吧
#然后加权的细节到底要怎么个加法呢？皮尔逊系数需要计算的嘛？
#我现在的问题就是想太多以至于现在没办法静下心来完成代码咯
def weighted_nn_predict(nodes_list, max_evals=10):
    
    num = len(nodes_list)
    #用于统计训练集输出的情况
    clf_train_pred = []
    #用于统计测试集输出的情况
    clf_test_pred = []
    #用于收集分类器
    clf_list = []
    #汇总训练集最后的输出
    train_pred_cnt = [0] * len(X_split_train)
    #汇总测试集最后的输出
    test_pred_cnt = [0] * len(X_split_test)
    
    for i in range(0, num):
        
        best_acc = 0.0
        best_model = 0.0
        
        for j in range(0, max_evals):
            
            clf = NeuralNetClassifier(lr = nodes_list[i]["lr"],
                                      optimizer__weight_decay = nodes_list[i]["optimizer__weight_decay"],
                                      criterion = nodes_list[i]["criterion"],
                                      batch_size = nodes_list[i]["batch_size"],
                                      optimizer__betas = nodes_list[i]["optimizer__betas"],
                                      module = create_module(nodes_list[i]["input_nodes"], nodes_list[i]["hidden_layers"], 
                                                          nodes_list[i]["hidden_nodes"], nodes_list[i]["output_nodes"], nodes_list[i]["percentage"]),
                                      max_epochs = nodes_list[i]["max_epochs"],
                                      callbacks = [skorch.callbacks.EarlyStopping(patience=nodes_list[i]["patience"])],
                                      device = nodes_list[i]["device"],
                                      optimizer = nodes_list[i]["optimizer"]
                                      )
        
            init_module(clf.module, nodes_list[i]["weight_mode"], nodes_list[i]["bias"])
        
            clf.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong)) 
        
            metric = cal_nnclf_acc(clf, X_split_train, Y_split_train)
            print_nnclf_acc(metric)
        
            best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
        
        #将所有的模型搜集起来咯
        clf_list.append(best_model)
        
    #对于训练集开始预测咯
    #先输出每个模型的准确率
    #然后输出模型的皮尔逊系数
    #最后输出加权之后的平均准确率
    for i in range(0, num):
        print(cal_nnclf_acc(clf_list[i], X_split_train, Y_split_train))
        clf_train_pred.append(clf_list[i].predict(X_split_train.values.astype(np.float32)))
        
    #开始计算训练集的皮尔逊矩阵咯
    #python真的很神奇呢，直接传入一个list
    #list中仅含有一个numpy.ndarray元素未报错
    pearson_coeff = np.corrcoef(clf_train_pred)
    print(pearson_coeff)
    
    #现在开始统计训练集最终的输出咯
    for i in range(0, num):
        for j in range(0, len(X_split_train)):
            if(clf_train_pred[i][j]==1):
                train_pred_cnt[j]=train_pred_cnt[j]+1
    
    for i in range(0, len(X_split_train)):
        if(train_pred_cnt[i]>(num)/2):
            train_pred_cnt[i]=1
        else:
            train_pred_cnt[i]=0
            
    #计算在训练集上的准确率呢
    count = (train_pred_cnt == Y_split_train).sum()
    train_acc = count/len(Y_split_train)
    print("accuracy on the train dataset is:",train_acc)
    
    
    #现在开始计算在训练集上面的情况
    for i in range(0, num):
        print(cal_nnclf_acc(clf_list[i], X_split_test, Y_split_test))
        clf_test_pred.append(clf_list[i].predict(X_split_test.values.astype(np.float32)))
    
    pearson_coeff = np.corrcoef(clf_test_pred)
    print(pearson_coeff)
    
    for i in range(0, num):
        for j in range(0, len(X_split_test)):
            if(clf_test_pred[i][j]==1):
                test_pred_cnt[j]=test_pred_cnt[j]+1
    
    for i in range(0, len(X_split_test)):
        if(test_pred_cnt[i]>(num)/2):
            test_pred_cnt[i]=1
        else:
            test_pred_cnt[i]=0

    count = (test_pred_cnt == Y_split_test).sum()
    test_acc = count/len(Y_split_test)
    print("accuracy on the test dataset is:", test_acc)
    
    
#这个版本直接试一哈四个基模型带来的效果
def weighted_predict(max_evals=10):
    
    clf_list = []
    clf_train_pred = []
    train_pred_cnt = [0] * len(X_split_train)
    clf_test_pred = []
    test_pred_cnt = [0] * len(X_split_test)
    
    #首先用神经网络训练一哈模型咯，因为这个计算结果非常不稳定所以要多算几次    
    best_acc = 0.0
    best_model = 0.0
        
    for i in range(0, max_evals):
            
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module = create_module(best_nodes["input_nodes"], best_nodes["hidden_layers"], 
                                                         best_nodes["hidden_nodes"], best_nodes["output_nodes"], best_nodes["percentage"]),
                                  max_epochs = best_nodes["max_epochs"],
                                  callbacks = [skorch.callbacks.EarlyStopping(patience=best_nodes["patience"])],
                                  device = best_nodes["device"],
                                  optimizer = best_nodes["optimizer"]
                                  )
        
        init_module(clf.module, best_nodes["weight_mode"], best_nodes["bias"])
        #这边修改为训练集和测试集咯
        clf.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_split_train, Y_split_train)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
            
    clf1 = best_model
    #现在使用一哈各种各样的模型在训练集上进行训练
    clf2 = XGBClassifier()
    clf2.fit(X_split_train, Y_split_train)
    clf3 = KNeighborsClassifier()  
    clf3.fit(X_split_train, Y_split_train)
    clf4 = LogisticRegression(penalty='l2')  
    clf4.fit(X_split_train, Y_split_train)
    clf5 = RandomForestClassifier(n_estimators=8)  
    clf5.fit(X_split_train, Y_split_train)
    
    #将所有分类器都加入到list中呢
    clf_list.append(clf1)
    clf_list.append(clf2)
    clf_list.append(clf3)
    clf_list.append(clf4)
    clf_list.append(clf5)
    
    #计算训练集上面的得分咯
    print(cal_nnclf_acc(clf1, X_split_train, Y_split_train))
    print(clf2.score(X_split_train, Y_split_train))
    print(clf3.score(X_split_train, Y_split_train))
    print(clf4.score(X_split_train, Y_split_train))
    print(clf5.score(X_split_train, Y_split_train))
    
    clf_train_pred.append(clf1.predict(X_split_train.values.astype(np.float32)))
    clf_train_pred.append(clf2.predict(X_split_train))
    clf_train_pred.append(clf3.predict(X_split_train))
    clf_train_pred.append(clf4.predict(X_split_train))
    clf_train_pred.append(clf5.predict(X_split_train))

    num =len(clf_list)
    
    pearson_coeff = np.corrcoef(clf_train_pred)
    print(pearson_coeff)
    
    for i in range(0, num):
        for j in range(0, len(X_split_train)):
            if(clf_train_pred[i][j]==1):
                train_pred_cnt[j]=train_pred_cnt[j]+1
    
    for i in range(0, len(X_split_train)):
        if(train_pred_cnt[i]>(num)/2):
            train_pred_cnt[i]=1
        else:
            train_pred_cnt[i]=0
            
    count = (train_pred_cnt == Y_split_train).sum()
    train_acc = count/len(Y_split_train)
    print("accuracy on the train dataset is:",train_acc)

    #计算在测试集上面的各种情况咯
    print(cal_nnclf_acc(clf1, X_split_test, Y_split_test))
    print(clf2.score(X_split_test, Y_split_test))
    print(clf3.score(X_split_test, Y_split_test))
    print(clf4.score(X_split_test, Y_split_test))
    print(clf5.score(X_split_test, Y_split_test))
    
    clf_test_pred.append(clf1.predict(X_split_test.values.astype(np.float32)))
    clf_test_pred.append(clf2.predict(X_split_test))
    clf_test_pred.append(clf3.predict(X_split_test))
    clf_test_pred.append(clf4.predict(X_split_test))
    clf_test_pred.append(clf5.predict(X_split_test))

    num =len(clf_list)
    
    pearson_coeff = np.corrcoef(clf_test_pred)
    print(pearson_coeff)
    
    for i in range(0, num):
        for j in range(0, len(X_split_test)):
            if(clf_test_pred[i][j]==1):
                test_pred_cnt[j]=test_pred_cnt[j]+1
    
    for i in range(0, len(X_split_test)):
        if(test_pred_cnt[i]>(num)/2):
            test_pred_cnt[i]=1
        else:
            test_pred_cnt[i]=0
            
    count = (test_pred_cnt == Y_split_test).sum()
    test_acc = count/len(Y_split_test)
    print("accuracy on the test dataset is:", test_acc)

#我找到了一个stacking的库咯，我感觉这个还挺好用的有点性感咯
def stacking_predict():
    
    #这里使用一个小trick：用局部变量计算全局准确率
    #使用stacking之后的模型轻松的达到了86%的准确率咯
    #这个让我在神经网络上面应用似乎看到了希望
    #所以明天的工作就是神经网络应用以及源代码原理理解。
    #再然后尝试一下周志华的论文提供的方法不知道是否有用
    #X_split_train = X_train_scaled
    #Y_split_train = Y_train
    
    xgb = XGBClassifier()
    knn = KNeighborsClassifier()  
    lr = LogisticRegression(penalty='l2')  
    rfc = RandomForestClassifier(n_estimators=8)
    
    sclf1 = StackingCVClassifier(classifiers=[knn, rfc], meta_classifier=xgb)
    sclf1.fit(X_split_train.values, Y_split_train.values)
    print("sclf1 on the train dataset.", sclf1.score(X_split_train.values, Y_split_train.values))
    print("sclf1 on the test dataset.", sclf1.score(X_split_test.values, Y_split_test.values))
    print()
    
    sclf2 = StackingCVClassifier(classifiers=[knn, lr], meta_classifier=xgb)
    sclf2.fit(X_split_train.values, Y_split_train.values)
    print("sclf2 on the train dataset.", sclf2.score(X_split_train.values, Y_split_train.values))
    print("sclf2 on the test dataset.", sclf2.score(X_split_test.values, Y_split_test.values))
    print()
    
    sclf3 = StackingCVClassifier(classifiers=[rfc, lr], meta_classifier=xgb)
    sclf3.fit(X_split_train.values, Y_split_train.values)
    print("sclf3 on the train dataset.", sclf3.score(X_split_train.values, Y_split_train.values))
    print("sclf3 on the test dataset.", sclf3.score(X_split_test.values, Y_split_test.values))
    print()
    
    sclf4 = StackingCVClassifier(classifiers=[lr, knn, rfc], meta_classifier=xgb)
    sclf4.fit(X_split_train.values, Y_split_train.values)
    print("sclf4 on the train dataset.", sclf4.score(X_split_train.values, Y_split_train.values))
    print("sclf4 on the test dataset.", sclf4.score(X_split_test.values, Y_split_test.values))
    print()
    
    sclf5 = XGBClassifier()
    sclf5.fit(X_split_train.values, Y_split_train.values)
    print("sclf5 on the train dataset.", sclf5.score(X_split_train.values, Y_split_train.values))
    print("sclf5 on the test dataset.", sclf5.score(X_split_test.values, Y_split_test.values))
    print()
    
#明天上班先实现这个东西再说吧
def stacking_nn_predict(nodes_list, max_evals):
    
    num = len(nodes_list)
    
    clf_list = []
    
    for i in range(0, num):
        
        best_acc = 0.0
        best_model = 0.0
        
        for j in range(0, max_evals):
            
            clf = NeuralNetClassifier(lr = nodes_list[i]["lr"],
                                      optimizer__weight_decay = nodes_list[i]["optimizer__weight_decay"],
                                      criterion = nodes_list[i]["criterion"],
                                      batch_size = nodes_list[i]["batch_size"],
                                      optimizer__betas = nodes_list[i]["optimizer__betas"],
                                      module = create_module(nodes_list[i]["input_nodes"], nodes_list[i]["hidden_layers"], 
                                                          nodes_list[i]["hidden_nodes"], nodes_list[i]["output_nodes"], nodes_list[i]["percentage"]),
                                      max_epochs = nodes_list[i]["max_epochs"],
                                      callbacks = [skorch.callbacks.EarlyStopping(patience=nodes_list[i]["patience"])],
                                      device = nodes_list[i]["device"],
                                      optimizer = nodes_list[i]["optimizer"]
                                      )
        
            init_module(clf.module, nodes_list[i]["weight_mode"], nodes_list[i]["bias"])
        
            clf.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong)) 
        
            metric = cal_nnclf_acc(clf, X_split_train, Y_split_train)
            print_nnclf_acc(metric)
        
            best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
        
        #将所有的模型搜集起来咯
        clf_list.append(best_model)
        
    meta_clf = clf_list[-1]
    clf_list.pop()
    sclf = StackingCVClassifier(classifiers=clf_list, meta_classifier=meta_clf)
    
    sclf.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong))
    print(sclf.score(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong)))
    
#现在直接利用经验参数值进行搜索咯，这样可以节约计算资源   
space = {"title":hp.choice("title", ["titanic"]),
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

space_nodes = {"title":["titanic"],
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


best_nodes = {"title":"titanic",
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

#现在需要划分一下训练集和测试集才能够看到加权之后是否模型得到提升
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.2)

"""
#可能下面的这种方式计算出的模型组合起来相关性太高，
#下面我试试别的模型看看能否找到更好的办法呢。
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

#这里需要注意一下的是，max_evals的取值必须大于nodes_list的数目
best_params = fmin(nn_f, space, algo=algo, max_evals=10, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

#根据超参搜索的结果创建模型咯
nodes_list = parse_trials(trials, space_nodes, 3)
#皮尔逊系数原来就是去中心的余弦计算
weighted_nn_predict(nodes_list, max_evals=2)
#最后输出的结果是这个样子的，这也太劲爆了吧
#这个分类器在集成之前的准确率是：
#0.8406285072951739
#0.8316498316498316
#0.8428731762065096
#0.6161616161616161
#0.8439955106621774
#[[1.         0.91605808 0.89599813 0.91098247 0.93584799]
# [0.91605808 1.         0.95204475 0.90790944 0.897972  ]
# [0.89599813 0.95204475 1.         0.88180885 0.88743012]
# [0.91098247 0.90790944 0.88180885 1.         0.94926097]
# [0.93584799 0.897972   0.88743012 0.94926097 1.        ]]
#Backend Qt5Agg is interactive backend. Turning interactive mode on.
#0.5454545454545454
#我就说这个结果太离谱了，原来是我的代码存在一些问题咯。
#皮尔逊系数和协方差有点关系，绝对值越大相关性越高，相关性可为正负。
#现在的结果是0.8395061728395061，出现了轻微的下降这个还可以理解
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#几次的计算结果都显示神经网络完爆xgboost之外的模型咯
weighted_predict()
#0.8412921348314607
#0.8595505617977528
#0.848314606741573
#0.8202247191011236
#0.8820224719101124
#[[1.         0.86391703 0.81326472 0.87929954 0.77959039]
# [0.86391703 1.         0.77133681 0.80119485 0.86996378]
# [0.81326472 0.77133681 1.         0.78243969 0.80119485]
# [0.87929954 0.80119485 0.78243969 1.         0.70464347]
# [0.77959039 0.86996378 0.80119485 0.70464347 1.        ]]
#accuracy on the train dataset is: 0.8567415730337079
#
#0.8268156424581006
#0.7877094972067039
#0.8156424581005587
#0.7932960893854749
#0.7653631284916201
#[[1.         0.79496308 0.75782163 0.8398138  0.77132498]
# [0.79496308 1.         0.78776009 0.69245241 0.88028218]
# [0.75782163 0.78776009 1.         0.74791196 0.81755854]
# [0.8398138  0.69245241 0.74791196 1.         0.68714978]
# [0.77132498 0.88028218 0.81755854 0.68714978 1.        ]]
#accuracy on the test dataset is: 0.7932960893854749
"""

"""
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

sclf.fit(X, y)
print(X)
print(len(X))
print(y)
print(len(y))
print(sclf.score(X,y))

sclf.fit(X_split_train.values, Y_split_train.values)
print(len(X_split_train))
print(len(Y_split_train))
print(sclf.score(X_split_train.values, Y_split_train.values))
"""

"""
#这个实验确实是说明了stacking以后的模型比单模型总体而言
#通过多次运行程序并比较 sclf4和sclf5可以得到结论：
#在训练集上稍弱，但是在测试集上稍强且稳定
#通过多次比较sclf4与sclf1、sclf2、sclf3、sclf4可以得到结论：
#在训练集上微弱优势，但在测试集上面优势明显，看来无脑加分类器就完事了？
stacking_predict()
#sclf1 on the train dataset. 0.8412921348314607
#sclf1 on the test dataset. 0.8044692737430168
#
#sclf2 on the train dataset. 0.8328651685393258
#sclf2 on the test dataset. 0.7988826815642458
#
#sclf3 on the train dataset. 0.8398876404494382
#sclf3 on the test dataset. 0.8044692737430168
#
#sclf4 on the train dataset. 0.8553370786516854
#sclf4 on the test dataset. 0.8212290502793296
#
#sclf5 on the train dataset. 0.8525280898876404
#sclf5 on the test dataset. 0.8212290502793296
"""

start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=5, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

nodes_list = parse_trials(trials, space_nodes, 3)
stacking_nn_predict(nodes_list, max_evals=2)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
