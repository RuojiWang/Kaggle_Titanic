#coding=utf-8
#����汾Ӧ�þ���ͨ��������ʵ�鷢�ֵڶ���Ч����õ�Ӧ���ǽ���lr�ĳ�������
#������ʹ��tpot����ʹ�õڶ���ı�Ҷ˹�Ż�������������������Ż���lr��ѹ����
#Ȼ���ڹ۲��������й����з����˵��ڵ�ѵ�������д��ڹ���ϵ����Ⲣ������һ�ֽ������
#�������⣬����stacking����������ģ�͵������Լ����ڵ������Ĵ����ȵȾ����ȶ��𰸣��ڶ��㲻������������
#�ҽ���Щʵ��Ľ���γ�������汾�Ĵ��뿩����������汾�Ľ���ύ����
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
#��������kfold��ʵ��k�۽���Ĺ��ܣ�����ÿ�ε�indice����������Ϊshuffle��Ĭ��δ��
#Ȼ�����StratifiedKFold�Ƿ���k�۽���ĵ�������ÿ��ͨ�����������ؽ������������Ϊshuffle
#���ߵ���������ǰ�߷���indice���������б����ֱ�ӷ��ص���������Ȼ����һ�ݴ������ַ�ʽ���е��������ǲ����
#from sklearn.model_selection import KFold,StratifiedKFold

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
    
#�����mode�����pandas.core.series.Series�����ĵ�һ��ֵ�������ж��������
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#��data_test�е�fareԪ����ȱʧ�Ĳ������Ѿ����������ݵ���λ��������
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

#�������˵������ǹ��״�Ʊ��ԭ����Ӣ�����������û������˵����
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
#print(df)
df_ticket = df.index.values          #����Ʊ��Ʊ��
tickets = data_train.Ticket.values   #���еĴ�Ʊ
#print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #�������д�Ʊ���ڹ���Ʊ�����Ϊ1������Ϊ0
    result.append(ticket)
    
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          #����Ʊ��Ʊ��
tickets = data_train.Ticket.values   #���еĴ�Ʊ

result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #�������д�Ʊ���ڹ���Ʊ�����Ϊ1������Ϊ0
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
#�Ҿ���ѵ�����Ͳ��Լ���Ҫ��һ������������ţ�����ע�͵���ԭ����X_train���������ſ�
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

#���������ʽ�޸Ĵ�������򵥶���ȫ��Ӱ����С�ķ�ʽ�˰�
#����ÿ�εõ���stacked_train��һ�����Ա����best_model��û����ô������
def save_stacked_dataset(stacked_train, stacked_test, title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "wb")
    pickle.dump([stacked_train, stacked_test], files)
    files.close()
    
def load_stacked_dataset(title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "rb")
    stacked_train, stacked_test = pickle.load(files)
    files.close()
    
    return stacked_train, stacked_test

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

#����ʱ��������������ǲ����е�ʱ��Ӧ���ظ������Ա���©����ѵĳ��Σ���
#�����һ����Ƕȿ���������⣺���������������ȫ�����ŵ���̫���׳�����Ҳ���а�
#���Դ�����Ƕȳ����Ļ����Ҿ��������ַ�ʽ��������ʵҲ���е���İɡ�
#OK�������µ������ֲ����ˣ����ڵ������ǣ�nn_f�ĺ�����֧�ִ��ݲ���
#Ϊ��֧�����γ����������������ַ�ʽʵ��һ�ַ�ʽ��д��������nn_f�ĺ�����
#����һ�ַ�ʽ����ִ��nn_f֮ǰ����ѵ�����ı���ָ������һ��ѵ������vans��
#�Ҿ��ôӴ����ά���Եȷ�����˵Ӧ�ú�һ�ַ�ʽ�Ǹ�ǡ���ķ�ʽ�ɣ�����Ҳ��Ҫ�ع����뿩
#�������������������������£�����nn_f�����ܹ�ʵ��ͨ�ã����������Ҫд����ѽ
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
    
    #��ߵ�columns���Լ������е�ѡ�񲿷�
    #��������һ�²��Ӻ�ȫ��֮��������أ�
    #X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], stacked_train, Y_train, columns=[i for i in range(0, stacked_train.columns.size)])
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], stacked_train, Y_train, columns=[])
    
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              #Ϊ�˲������´���space,space_nodes���������д����
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
    
    #����ӵ���ЩԪ�����ڿ���ģ�͵Ľṹ
    best_nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[0]["misc"]["vals"]["input_nodes"][0]]
    best_nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[0]["misc"]["vals"]["hidden_layers"][0]]
    best_nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[0]["misc"]["vals"]["hidden_nodes"][0]]
    best_nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[0]["misc"]["vals"]["output_nodes"][0]]
    best_nodes["percentage"] = space_nodes["percentage"][trials_list[0]["misc"]["vals"]["percentage"][0]]

    return best_nodes

#�ҷ�������������һ��BUG���������˹ֲ���ûɶ�ý��
def parse_trials(trials, space_nodes, num):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    #nodes = {}nodes�����������ôÿ�θ���֮����һ���Ŀ�
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

#���ѡ�����ģ�͵�ʱ����ڹ���ϵķ���
def nn_model_train(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #����������ģ�ͳ�ʼ����dropout�ȵ����⵼�����粻���ȶ�
    #����������İ취���Ƕ��ظ����㼸�Σ�ѡ�����п��׵�ģ��
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

#�Ҿ�������һ���ķ�ʽ��Сģ��ѡ��ʱ����ܴ����Ĺ���Ϸ��հ�
#Ϊ�˲��ı�ԭ�������Ľӿڻ�������С�޸Ĵ��۵ķ�ʽ�޸Ĵ������뵽������İ취��
#������޸ķ�ʽ����֮ǰ�뵽���޸ķ�ʽ�о��ϻ�Ҫ����һЩ���ء���
def nn_model_train_validate(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #�Ҿ���0.12�������е���ˣ����кܶ�����û�õ��أ��о�����������Ӧ�û��һЩ�İɣ�
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.05, stratify=Y_train)
    #����������ģ�ͳ�ʼ����dropout�ȵ����⵼�����粻���ȶ�
    #����������İ취���Ƕ��ظ����㼸�Σ�ѡ�����п��׵�ģ��
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
        clf.fit(X_split_train.astype(np.float32), Y_split_train.astype(np.longlong))
            
        metric = cal_nnclf_acc(clf, X_split_test, Y_split_test)
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
        #�������ݼ�
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

def get_oof_validate(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_acc = []
    valida_acc = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #�������ݼ�
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_acc = nn_model_train_validate(nodes, X_split_train, Y_split_train, max_evals)
        
        #��������������ģ�͵�ѵ��������֤������Ľ����
        #�����׺������ѵ�����̵����һ������ص�
        #�������������϶��ǲ�һ���ģ�
        #��һ������͵ڶ�������������������ģ�ͺ���ͨģ����ѵ������������
        #�ڶ�������͵���������������������ģ����ѵ��������֤����������
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

def stacked_features_validate(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_validate(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#���ѡ�����ģ�͵�ʱ����ڹ���ϵķ���
def nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0

    #���Ѿ�����ݴ����best_nodes["title"]��ԭ����titanic��Ϊstacked_titanic��Ϊ�°汾
    if (exist_files(best_nodes["title"])):
        #��������ʱ������stakced_train�Լ�stacked_test��
        best_model = load_best_model(best_nodes["title"]+"_"+str(len(nodes_list)))
        best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_train.values)
         
    for i in range(0, max_evals):
        
        #��߲��Ǻ�����nn_model_train��������ĺ�������
        #��Ϊ������Ĵ��뻹�漰��Ԥ����������ⲻ���޸�
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
            #����汾��best_model������ȫ�ֵİ汾�������ǿ�ɭ�ء���
            save_best_model(best_model, best_nodes["title"]+"_"+str(len(nodes_list)))
            Y_pred = best_model.predict(stacked_test.values.astype(np.float32))
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred
   
#lrû�г�����������û�н��й�cv��ô���ܻ�ȡ�úóɼ��أ� 
def lr_stacking_predict(stacked_train, Y_train, stacked_test, max_evals=50):
    
    best_acc = 0.0
    best_model = 0.0
       
    #���ﲢ����Ҫ������ѵ�ģ�Ͱɣ�ֻ��Ҫ��stacked_train֮������ݼ�¼����������
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        #����ǲ�����Ҫ����һЩ����������ػ����������أ���
        clf = LogisticRegression()        
        clf.fit(stacked_train, Y_train)
        
        metric = cal_nnclf_acc(clf, stacked_train.values, Y_train.values)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        if (flag):
            #����汾��best_model������ȫ�ֵİ汾�������ǿ�ɭ�ء���
            save_best_model(best_model, best_nodes["title"]+"_"+str(len(nodes_list)))
            Y_pred = best_model.predict(stacked_test.values.astype(np.float32))
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred

#lr�����˳�������ѡ����õĽ������Ԥ�⿩ 
def lr_stacking_cv_predict(stacked_train, Y_train, stacked_test, max_evals=2000):
    
    clf = LogisticRegression()
    param_dist = {"penalty": ["l1", "l2"],
                  "C": np.linspace(0.001, 100000, 10000),
                  "fit_intercept": [True, False],
                  #"solver": ["newton-cg", "lbfgs", "liblinear", "sag"]
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=max_evals)
    random_search.fit(stacked_train, Y_train)
    best_acc = random_search.best_estimator_.score(stacked_train, Y_train)
    lr_pred = random_search.best_estimator_.predict(stacked_test)

    save_best_model(random_search.best_estimator_, best_nodes["title"]+"_"+str(len(nodes_list)))
    Y_pred = random_search.best_estimator_.predict(stacked_test.values.astype(np.float32))
            
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(best_nodes["path"], index=False)
    print("prediction file has been written.")
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return random_search.best_estimator_, Y_pred

def tpot_stacking_predict(stacked_train, Y_train, stacked_test, generations=100, population_size=100):
    
    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity = 2)
    tpot.fit(stacked_train, Y_train)
    best_acc = tpot.score(stacked_train, Y_train)
    Y_pred = tpot.predict(stacked_test)
    best_model = tpot
         
    save_best_model(best_model.fitted_pipeline_, best_nodes["title"]+"_"+"tpot")
    Y_pred = best_model.predict(stacked_test)
            
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(best_nodes["path"], index=False)
    print("prediction file has been written.")
            
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred
    
#����ֱ�����þ������ֵ�������������������Խ�Լ������Դ
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

#��ʵ������Ҫbest_nodes��Ҫ��Ϊ�˿��ٲ���
#��Ȼÿ�γ���������best_nodesЧ��̫���˰�
best_nodes = {"title":"stacked_titanic",
              "path":"C:/Users/1/Desktop/Titanic_Prediction.csv",
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
#�����ģ�;�Ȼȡ����85.29%����ȷ�ʣ�������ǿ�����ϣ���ˣ�����stacking��������ѽ
#the best accuracy rate of the model on the whole train dataset is: 0.8529741863075196
#�еĵط���.values�еĵط���û������о������һ��Ƕ��ð�
#��ʵ�����Ӧ��֪���ģ�ֱ�Ӱ�stacked_train֮��ı��df��
algo = partial(tpe.suggest, n_startup_jobs=10)
#��������ظ����ӳ��νڵ�����Ȼû�иı�Ү��5���ڵ�������Ч��
#�о�ֱ�������ظ��Ĵ������ܹ��õ�������������˼��������һ�������Ƚ����ް�
#���Ӽ�������������Ǻ����ԣ��������ӽڵ���Ŀ���������е�����Ŷ
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes, 
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 10)
stacked_trials = Trials()
#��Ȼ����Ƿ���Ϊ�����汾������Щ����Ҫ�˰�
#X_train_f = stacked_train
#Y_train_f = Y_train
#��������д�����У���Ϊ������Ŀ���ʹ����ǰ��trials������޸��˾Ͳ����˰�
#space["input_nodes"]=hp.choice("input_nodes", [len(nodes_list)])
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
#�����⺯��һֱ���������Һܶ��ʱ���֪����֮ǰ�洢��stacked_titanic_best_model������
nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 20)
"""

"""
#��������Ҫ������Ķ����������������˰ɣ����Բο��������õ�����
start_time = datetime.datetime.now()
algo = partial(tpe.suggest, n_startup_jobs=10)

nodes_list = [best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 20)

stacked_trials = Trials()
#������ֿ����ٿ���һ��ɣ����˽ڵ���Ŀ�������ֿ��Բ������ˡ�
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=20, trials=stacked_trials)
print_best_params_acc(stacked_trials)
best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 20)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#����ı���汾�Ĵ������ǵ�ͨʵ���˰�
start_time = datetime.datetime.now()

nodes_list = [best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 5)

stacked_trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=5, trials=stacked_trials)

best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 5)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

stacked_train, stacked_test = load_stacked_dataset("stacked_titanic")
best_model = load_best_model("stacked_titanic_2")
print(cal_nnclf_acc(best_model, stacked_train.values, Y_train.values))
"""

"""
#����ı���汾�Ĵ������ǵ�ͨʵ���˰�
#�������ݴ�������һ�������İ汾�ɣ���������Ĵ������һ������
#�ڶ���һ�㲻��ʹ�ñȽϼ򵥵�ģ�������ܹ���ֹ����Ͽ�
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=7000, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

nodes_list = parse_trials(trials, space_nodes, 9)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
#              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 25)

stacked_trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=100, trials=stacked_trials)

best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 50)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
stacked_train, stacked_test = load_stacked_dataset("stacked_titanic")
best_model = load_best_model("stacked_titanic_2")
print(cal_nnclf_acc(best_model, stacked_train.values, Y_train.values))
"""

"""
#����Ĵ�������7000�β��Ե�ǰϷ�ɣ������Ϳ������ǵ�ִ�д���㿩
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=10, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

nodes_list = parse_trials(trials, space_nodes, 9)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
#              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 10)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

lr_stacking_predict(stacked_train, Y_train, stacked_test, 10)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#����Ĵ���Ӧ������������ʵ�ֵİ汾��
#����汾�ڵڶ��㻹��Ҫʹ���߼��ع鿩
#lr�ڴ�����б��ֵĽ��ȷʵ̫��⣬
#�о�ֻ�ܹ��ڵڶ���ʹ��tpot����
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=7000, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

nodes_list = parse_trials(trials, space_nodes, 9)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
#              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 50)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

lr_stacking_predict(stacked_train, Y_train, stacked_test, 100)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#��Ŀǰ�Ľ�����ԣ�����Ĵ���Ӧ����Ŀǰ�ܹ�ȡ�õ����Ч��֮һ
#��ݴ�����Ҫ��������Ǽ����������ˣ��漰�������γ�������
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=7000, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

#nodes_list = parse_trials(trials, space_nodes, 9)
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 70)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

stacked_trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_stacked_params = fmin(nn_stacking_f, space, algo=algo, max_evals=7000, trials=stacked_trials)

best_nodes = parse_nodes(stacked_trials, space_nodes)
save_inter_params(stacked_trials, space_nodes, best_nodes, "stacked_titanic")

nn_stacking_predict(best_nodes, nodes_list, stacked_train, Y_train, stacked_test, 1000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#������ݴ����ģ���ܹ�����ڶ��γ�������
#�����˵���ǽ���һ�ε�����֮��stacking֮������
#��Ϊ�µ�������Ȼ������tpot�ķ�ʽ�������������������
#�������ĺô�����һ�����ܹ���Լ�ڶ��γ���������ʱ��
#���ұ����˵�һ��stacking֮ǰ���������̵�ʱ�俩
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=2, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

#nodes_list = parse_trials(trials, space_nodes, 9)
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes,
              best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 1)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

tpot_stacking_predict(stacked_train, Y_train, stacked_test, generations=3, population_size=3)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

#������ݴ����ģ������Ŀǰ���µ�ʵ�����
#��Ŀǰ��ʵ������Ӧ�ÿ��������յİ汾����������������ٳ���֮ǰ��ǰ23%�ĳɼ���
#��Ҫ�ɹ����ڷ����˵ڶ���ʹ����������������߼��ع���֤��Ч�����
#���µķ��֣����ڵ��ƺ���ʹ�õ�Խ�����������Խ�ã��������ڸ�Ϊ45���ڵ㣿��
#Ȼ��ʵ����stacked_features_validateϵ�к�����������������˼·�����ǣ�
#�����������ݣ�����5%��������ѵ����ֻ���ڹ������֤&ģ��ѡ�����early stoppingӦ�ò���ܹ���ֹ������˰�
#���������ʵ�ֵ�ʱ�򻹿����˾�����ҪӰ��֮ǰ��stacked_featuresϵ�к����Ľӿ����
#stacked_features_validateϵ�к����ĵڶ�������������֤���ֵ��20����һ��������ֵ���������ͽڵ���Ŀ����Խ��Խ�á�
#��һ��������ֵ��������Խ���ܹ�ʹ�õ������ݾ�Խ�����׼ȷ��һ������������ǽڵ���Ŀ��ģ������Խ��Ϊë��Խ�����ء���
#��������Ϊģ������Խ���ʱ���߼��ع��ܹ��õ�������ά��Խ��Ӷ�����Խ����Խ�ȶ�����
#�Ҿ��������뿪��ʱ�������һ��50�۵Ľ�����֤����45���ڵ㡣���о�Ӧ�û�ܾ����İɡ��������ʾ�ǳ�������ѽ���Ҷ��۾������ˡ���
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(nn_f, space, algo=algo, max_evals=7, trials=trials)

best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")

#nodes_list = parse_trials(trials, space_nodes, 9)
nodes_list = [best_nodes, best_nodes]
stacked_train, stacked_test = stacked_features_validate(nodes_list, X_train_scaled, Y_train, X_test_scaled, 5, 5)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")

lr_stacking_cv_predict(stacked_train, Y_train, stacked_test, 2000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

#�����Ƕ����������Ľ�����������ܽ���չ������
#��0�����ڻ���ɶ�취���Խ�һ������kaggle��׼ȷ���ء����ҵ�һ���취���޸�stackingʱ�����������
#OK��ȷʵ�Ǹ����еİ취������������ģ����ĿҲ����Ŷ��������ԭ�����û�취����ѧ�Ͻ��ͣ�����
#��1���������ģ���ں��أ���������˺ܾþ�ֻ�ҵ���stacking����Ļ����û�У�
#ֻ��һƪ��־�������ĸо����뿴�ض���δ���ܿ�������ʱ��Ҳ�����˲������̽��������
#2���ǲ������ﻹ���Խ��н�����֤��������ʹ������ν�һ�����ѽ�� 
#������֤�����õ���ÿ������ӳ���ѡ��ڵ㵽�ڵ�ѡ����֮�����stacking��
#֮ǰ��ģ�ͷ������ܲ�����Ҫ����Ϊѡ����ѵ�ģ��ʱ������ˣ���������������ר������֤֮�󷺻����ܵõ���������
#�ҷ������������stacking�ĺô����ѽ�����ٲ�����ģ��������ģ�ͽ����Ż�������һ��һ���Ķ�ģ�ͽ����Ż�����
# ����ʹ������������޸���������ûɶ�����������ˣ�����Ҫ�����������ݽ�����֤����Ȼ��֤�ķ�ʽ������early stopping��
#���Ƕ��ѵ��ģ��ѡ������ʱearly stopping��dropout��ȫ���ã���򵥵İ취��������5%������������֤����
#��Ȼ������Ϊ�ļ���һЩ����Ȼ����Խ����������ݲ��뽻����֤�����������ȵĳ�����Щ����Ҫȷ���Ƚ��鷳��
#������ֹ����ϵķ�������˵������ҪAPI��֧�ְɣ�pytorch��API��ʵ֧�����������Ҫ����torch.optim.Adam�ڵĲ�����
#���ո�ȥ�������ڲ���������ʵ��֮ǰ�Ľڵ㳬��������ʱ���Ѿ���������������ˣ����Ե��ڵ�ģ��ѵ����ʱ����������ģ�
#���Ժ��������ʱֻ�������ڵİ취����ʵ���ڵ������Ѿ�������������֤+��������֤���Ѿ������ˡ���ģ�͵�ѵ��һ����ʹ�ó�ȥĳһ��֮����������ݿ���

#�Ժ���Բ��õĸ��õ�ϸ�ڣ�
#��1��������ɾ����Ⱥ�㿩