#coding=utf-8
#这个版本主要对于xgboost的模型进行超参搜索的吧
#但是我实话实说呀，我觉得这个实验的意义不大的吧
#因为就是以后做stacking也不一定需要xgboost的吧
#实现这个仅有的意义就在于知道了xgboost一个小时能够超参搜索1000次
#然后超参搜索了10000/1000次的结果居然才是85.1%所以xgboost也就这样了吧
import os
import sys
import random
import pickle
import datetime
import numpy as np
import pandas as pd
from astropy.modeling.tests.test_models import create_model
sys.path.append("D:\\Workspace\\Titanic")

from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

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

def cal_xgbclf_acc(clf, X_train, Y_train):
    
    return clf.score(X_train, Y_train)

def print_xgbclf_acc(acc):
    
    print("the accuracy rate of the model on the whole train dataset is:", acc)
  
def print_best_params_acc(trials):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    
    trials_list.sort(key=lambda item: item["result"]["loss"])
    
    print("best parameter is:", trials_list[0])
    print()
    
def exist_files(title):
    
    return os.path.exists(title+"_best_xgb_model.pickle")
    
def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_xgb_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params(title):
  
    files = open(str(title+"_xgb_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes
    
def save_best_model(best_model, title):
    
    files = open(str(title+"_best_xgb_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()
    
def load_best_model(title):
    
    files = open(str(title+"_best_xgb_model.pickle"), "rb")
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
    
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] +=  random.gauss(mean, std)

    return X_noise_train, Y_train

"""
#我反复确认了几次参数以及范围我觉得sklearn中的xgboost应该就是这些参数了吧
#我之前好像用过其他版本的xgboost，他们可能有些其他参数但是应该不会对结果影响太大
#果然colsample_bytree、subsample这些参数范围都是在[0, 1]之间呢
xgb = XGBClassifier(colsample_bytree=2.0)
xgb.fit(X_train_scaled, Y_train)
"""

def xgb__f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("max_depth", params["max_depth"])
    print("learning_rate", params["learning_rate"])
    print("n_estimators", params["n_estimators"])
    print("gamma", params["gamma"])
    print("min_child_weight", params["min_child_weight"])
    print("max_delta_step", params["max_delta_step"])
    print("subsample", params["subsample"])
    print("colsample_bytree", params["colsample_bytree"])
    print("reg_lambda", params["reg_lambda"])
    #print("early_stopping_rounds", params["early_stopping_rounds"])
        
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    
    xgb = XGBClassifier(max_depth = params["max_depth"], 
                        learning_rate = params["learning_rate"],
                        n_estimators = params["n_estimators"],
                        gamma = params["gamma"],
                        min_child_weight = params["min_child_weight"],
                        max_delta_step = params["max_delta_step"],
                        subsample =  params["subsample"],
                        colsample_bytree = params["colsample_bytree"],
                        reg_lambda = params["reg_lambda"],
                        #early_stopping_rounds = params["early_stopping_rounds"],
                        )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    metric = cross_val_score(xgb, X_noise_train, Y_noise_train, cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric
    
def parse_space(trials, space_nodes, best_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]
    best_nodes["gamma"] = space_nodes["gamma"][trials_list[0]["misc"]["vals"]["gamma"][0]]
    best_nodes["min_child_weight"] = space_nodes["min_child_weight"][trials_list[0]["misc"]["vals"]["min_child_weight"][0]]
    best_nodes["max_delta_step"] = space_nodes["max_delta_step"][trials_list[0]["misc"]["vals"]["max_delta_step"][0]]
    best_nodes["subsample"] = space_nodes["subsample"][trials_list[0]["misc"]["vals"]["subsample"][0]]
    best_nodes["colsample_bytree"] = space_nodes["colsample_bytree"][trials_list[0]["misc"]["vals"]["colsample_bytree"][0]]
    best_nodes["reg_lambda"] = space_nodes["reg_lambda"][trials_list[0]["misc"]["vals"]["reg_lambda"][0]]
    #best_nodes["early_stopping_rounds"] = space_nodes["early_stopping_rounds"][trials_list[0]["misc"]["vals"]["early_stopping_rounds"][0]]

    return best_nodes
    
def predict(best_nodes, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0

    if (exist_files(best_nodes["title"])):
        best_model = load_best_model(best_nodes["title"])
        best_acc = cal_xgbclf_acc(best_model, X_train_scaled, Y_train)
         
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        xgb = XGBClassifier(max_depth = best_nodes["max_depth"], 
                            learning_rate = best_nodes["learning_rate"],
                            n_estimators = best_nodes["n_estimators"],
                            gamma = best_nodes["gamma"],
                            min_child_weight = best_nodes["min_child_weight"],
                            max_delta_step = best_nodes["max_delta_step"],
                            subsample =  best_nodes["subsample"],
                            colsample_bytree = best_nodes["colsample_bytree"],
                            reg_lambda = best_nodes["reg_lambda"],
                            #early_stopping_rounds = best_nodes["early_stopping_rounds"],
                            )
        
        xgb.fit(X_train_scaled, Y_train) 
        metric = cal_xgbclf_acc(xgb, X_train_scaled, Y_train)
        print_xgbclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(xgb, metric, best_model, best_acc)
    
        if (flag):
            save_best_model(best_model, best_nodes["title"])
            Y_pred = best_model.predict(X_test_scaled)
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    
space = {"title":hp.choice("title", ["titanic"]),
         "path":hp.choice("path", ["C:/Users/win7/Desktop/Titanic_Prediction.csv"]),
         "mean":hp.choice("mean", [0]),
         "std":hp.choice("std", [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]),
         
         "max_depth":hp.choice("max_depth", [3, 4, 5, 6, 7, 8, 9, 10]),     
         "learning_rate":hp.choice("learning_rate", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                                                     0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                                                     0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
                                                     0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]),
         "n_estimators":hp.choice("n_estimators", [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 300, 500]),
         "gamma":hp.choice("gamma", [0, 0.05, 0.10, 0.15, 0.20]),
         "min_child_weight":hp.choice("min_child_weight", [0, 1, 2, 3, 4, 5]),
         "max_delta_step":hp.choice("max_delta_step", [0, 1, 2, 3, 4, 5]),
         "subsample":hp.choice("subsample", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
         "colsample_bytree":hp.choice("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
         "reg_lambda":hp.choice("reg_lambda", [0.1, 0.5, 1.0, 5.0, 50, 500, 5000]),
         #"early_stopping_rounds":hp.choice("early_stopping_rounds", [None])
         }

space_nodes = {"title":["titanic"],
               "path":["C:/Users/win7/Desktop/Titanic_Prediction.csv"],
               "mean":[0],
               "std":[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
                
               "max_depth":[3, 4, 5, 6, 7, 8, 9, 10],     
               "learning_rate":[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                                0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                                0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
                                0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40],
               "n_estimators":[70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 300, 500],
               "gamma":[0, 0.05, 0.10, 0.15, 0.20],
               "min_child_weight":[0, 1, 2, 3, 4, 5],
               "max_delta_step":[0, 1, 2, 3, 4, 5],
               "subsample":[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
               "colsample_bytree":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               "reg_lambda":[0.1, 0.5, 1.0, 5.0, 50, 500, 5000],
               #"early_stopping_rounds":[None],
               }

best_nodes = {"title":"titanic",
              "path":"path",
              "mean":0,
              "std":0.1,
              "max_depth":5,     
              "learning_rate":0.10,
              "n_estimators":100,
              "gamma":0,
              "min_child_weight":0,
              "max_delta_step":0,
              "subsample":1.0,
              "colsample_bytree":1.0,
              "reg_lambda":1.0,
              #"early_stopping_rounds":None
              }

start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(xgb__f, space, algo=algo, max_evals=10000, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_space(trials, space_nodes, best_nodes)
save_inter_params(trials, space_nodes, best_nodes, "titanic")
trials, space_nodes, best_nodes = load_inter_params("titanic")

predict(best_nodes, max_evals=1000)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

"""
#不过这个例子也告诉我们说xgboost的参数可真是很少很简洁呀。
#而且这个也没有涉及到模型的class保存的问题咯,其实不需要保存的吧。
#我原本以为做完这个实验大概只需要一个小时其实大致花了三个小时
#所以说软件工程的三倍定律到这里依然能够存在。
#我觉得这些在做比赛过程中冒出来的想法还是值得去尝试一下的咯。
files = open("titanic_xgb_intermediate_parameters_2018-10-6154024.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
print(best_nodes)
#print(space_nodes)

files = open("titanic_best_xgb_model_2018-10-6154024.pickle", "rb")
best_model = pickle.load(files)
files.close()
best_acc = cal_xgbclf_acc(best_model, X_train_scaled, Y_train)
print(best_acc)
#最后的输出大致是这个样子的咯
#{'title': 'titanic', 'path': 'C:/Users/win7/Desktop/Titanic_Prediction.csv', 'mean': 0, 'std': 0.1, 'max_depth': 8, 'learning_rate': 0.39, 'n_estimators': 150, 'gamma': 0.1, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_lambda': 50}
#0.8518518518518519
"""