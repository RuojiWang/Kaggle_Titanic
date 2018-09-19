#coding=utf-8
#这个版本的代码主要就是用于比较我的模型和TPOT模型来PK一下。。我估计我会死的很惨
#还好啦不要那么灰心丧气吧，至少。。你对于结果的判断还是很准的吧，你确实是被碾压。。
import sys
import random
import pickle
import numpy as np
import pandas as pd
sys.path.append("D:\\Workspace\\Titanic")
from Utilities1 import noise_augment_pytorch_classifier

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import skorch
from skorch import NeuralNetClassifier

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
    
    files = open("titanic_best_model.pickle", "rb")
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
                              max_epochs = 200,
                              optimizer=torch.optim.Adam,
                              callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
                              )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()
    return -metric

def display_search_progress(search_times, nn_f):
    
    print("search times:", search_times)
    return nn_f
    
def parse_space(trials, space_nodes, best_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
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
        
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" percentage progress have been made.")
        
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
                                  max_epochs = 200,
                                  optimizer=torch.optim.Adam,
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
                                  )
     
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        if flag:        
            
            save_best_model(best_model)
            Y_pred = best_model.predict(X_test.values.astype(np.float32))
            #将得到的预测结果写入到文件中去咯
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            output.to_csv("C:/Users/win7/Desktop/Titanic_Prediction.csv", index=False)
            print("prediction file has been written.")
        print()
    
    metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
    print("the best accuracy rate of the model on the whole train dataset is:", metric)

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
         "module":hp.choice("module", [module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
               module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]),
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
               "module":[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
                     module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
               }

best_nodes = {"mean":0,
              "std":0.1,
              "lr":0.0001,
              "optimizer__weight_decay":0.005,
              "criterion":torch.nn.NLLLoss,
              "batch_size":1,
              "optimizer__betas":[0.86, 0.999],
              "module":module1.weight_init1(),
             }

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=2, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_space(trials, space_nodes, best_nodes)
save_inter_params(trials, space_nodes, best_nodes)
trials, space_nodes, best_nodes = load_inter_params()

predict(best_nodes, max_evals=6)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

"""
# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)
"""

# Score on the training set was:0.9680999873581648
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.7500000000000001, min_samples_leaf=15, min_samples_split=11, n_estimators=100)),
    LogisticRegression(C=0.5, dual=True, penalty="l2")
)

exported_pipeline.fit(X_train_scaled, Y_train)

best_model = load_best_model()

result = pd.read_csv("C:/Users/win7/Desktop/gender_submission.csv")
Y_result = result["Survived"].values

acc1 = cal_nnclf_acc(best_model, X_test_scaled, Y_result)
print(acc1)
acc2 = cal_nnclf_acc(exported_pipeline, X_test_scaled, Y_result)
print(acc2)
#最后的结果居然是这个样子的，我感觉很心酸呢。。
#0.8421052631578947
#0.9019138755980861
