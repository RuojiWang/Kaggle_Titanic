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

#import skorch.net
#为什么用下面的方式import可以但是用上面的方式不行呢？
#使用from sys import argv, path那么可以直接使用path
#如果是采用import sys那么调用时候需sys.path
from skorch.net import NeuralNetClassifier
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

"""
#卧槽，这个版本的框架二有点问题，写完文件没有及时关闭导致只有计算完了之后才能写入文件
#这个问题我之前修改过，可能是因为没有拷贝到这边的机器上面的缘故吧，好在已经修复了这个问题。
#同时也终于实现了引用不同模块下的代码了，这样不会每次都到处复制粘贴代码啦。。
import sys
#sys.path.append("..\\..\\Titanic\\Utility.py")
#sys.path.append("D:\\Workspace\\Titanic")
sys.path.append("..\\Titanic")
from Utility import noise_augment_classifier

lr1 = LogisticRegression(class_weight={0:549/(549+342),1:342/(549+342)})
grid1 ={"penalty":["l1", "l2"],
        "random_state":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000, 10000]}
columns = [3, 4, 5, 6, 7, 8]
mean = [0, 0, 0, 0]
sigma =[0, 0.1, 1, 10]
clf = [lr1]
para = [grid1]
noise_augment_classifier(2, 2, X_train_scaled, Y_train, 0.3, lr1, grid1, mean, sigma, columns, 6000, n_iter=200)
"""

import sys
#sys.path.append("..\\..\\Titanic\\Utility.py")
sys.path.append("D:\\Workspace\\Titanic")
#sys.path.append("..\\Titanic")
#print(sys.path)
from Utilities1 import noise_augment_pytorch_classifier

#我觉得框架三应该就是直接的进行特征处理的部分吧，但是特征处理我可能还得思考一下咋做才行呀。
#模型层面我想控制的东西大概是：层数、每层的神经元个数、神经元的连接方式、dropout的程度。
#框架二的使用方式就是先确定最优噪声范围，然后通过给定几个最优噪声选项进行多次的battle，得到最优噪声的参数。
#最后通过给所有数据增加上这个噪声，然后超参搜索出一个模型，并对带预测样本进行预测。
#我这样做实际上是认为噪声是数据集的一部分，然后对这个数据集进行一次超参搜索的意思咯。
#说句实在的，我现在不知道这么做有没有道理，或者说我感觉从信息论的角度分析这个问题，感觉这么做好像没用？
#但是如何从信息论的角度看待dropout呢？可能dropout的做法也是没啥用的吧？
#我感觉自己的这方面的能力比较有限，以后还是少做点这种事情吧，这些东西还是主要从论文中获得感觉比较靠谱。
#或者换个角度，如果从数据集增强的角度还是有点道理的：我在实验怎么样去增强我的数据集让我的模型得到提升？
#好吧，我觉得短期内我好想说服了我自己，那么我暂时认为框架二还是比较有意思的。
#noise_augment_classifier函数的第五个参数和倒数第二个参数可以设置一些特殊的值，达到一些特殊的效果。
#所以说，即便是不进行数据集增强，框架二依然可以给我带来价值咯。
class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()

        #nn.Linear在linear.py中的__init__中赋初值
        #nn.Linear在linear.py中的reset_parameters赋值
        #reset_parameters使用均匀分布对权重和偏置进行赋值
        #https://zhuanlan.zhihu.com/p/38100200
        #高斯分布初始化的代码见上面链接，关键是如何apply???
        #首先，初始化的方式算作超参，必须是作为params的一部分咯
        #其次，我只能将初始化作为MyModule的方法
        #然后，这个方法必须返回self否则没有模型就无法超参搜索
        #最后，查阅torch.nn中init.py文件的初始化方式并做出选择即可
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

#    原来这个样子就能够让这个函数通过超参搜索咯。
#    def print1(self):
#        print("hello world")
#        return self

    #这个函数表示采用默认的初始化咯
    #我有点不知道到底设计几种初始化方案咯
    #初步考虑就设计下面四种方案就完事儿了吧
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
        #torch.nn.init.xavier_uniform_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        #torch.nn.init.xavier_uniform_(self.fc3.bias)
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
        #torch.nn.init.xavier_uniform_(self.fc1.bias.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        #torch.nn.init.xavier_uniform_(self.fc2.bias.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        #torch.nn.init.xavier_uniform_(self.fc3.bias.data)
        torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        #torch.nn.init.xavier_uniform_(self.fc4.bias.data)
        return self   
            
module1 = MyModule1()
module2 = MyModule2()

#这样子，我先把下面的东西理顺，再说和上面相结合的问题咯。
net = NeuralNetClassifier(
    module = module1,
    
    lr=0.1,
    #原来这里并不是GPU而是cuda呀S
    #好像pytorch版本框架二报错是因为cuda
    device="cuda",
    #device="cpu",
    max_epochs=10,
    #criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    #http://skorch.readthedocs.io/en/latest/user/neuralnet.html
    #上面的链接中的callback的部分讲解了如何传入自己的callback
    #然后查看net.py中的_default_callbacks函数能够知道如何实现函数
    #最后一个细节是如何打断训练，我现在查阅过max_epochs以及KeyboardInterrupt并不知如何代码打断训练
    #http://skorch.readthedocs.io/en/latest/callbacks.html上面有skorch.callbacks.EarlyStopping
    #然而我这里并不能够发现这个库，可能是库的版本的问题，我运行skorch的安装命令就直接更新到最新版本了
    #果然是更新到最新版本就可以了，从此妈妈再也不用担心我的max_epochs的需要枚举咯
    #按照时间推算的话，这个skroch版本的应该是最近一个月才出来的吧，花了这么多时间总算搞定这个问题了，看看代码就完事儿了
    #说真的这个东西让我自己实现起来还是挺麻烦的，主要是对于这些代码或者库的机制不够熟悉，比如说为何Checkpoint要实现_sink
    callbacks=[skorch.callbacks.EarlyStopping(patience=5)]
)

#确定流程是否达到预期而非仅仅是能够运行
#选择合适的参数进行数据集增强最终提交模型预测结果（可能涉及到模型的保存）
#以后将特征工程进行封装形成框架三吧，这样子的话数据过来自动特征工程，然后自动数据集增强并提交结果
params = {
    #这些算是模型训练参数
    #在net.py中详细的描述了这些参数的含义，值得注意的是lr是传给optimizer的
    #pytorch能够在不同的层设置不同的学习率，感觉真心是偏研究的工具，sklearn才是比赛工具
    #周志华、deep learning以及我收集到的材料里面找不到合适的经验参数，暂时就这样设置吧
    
    #这个epoch参数按照Pytorch或者skorch都必须作为超参咯。
    #这个epoch的值其实并不存在经验值吧，不同数据集取值范围不一样。
    #'max_epochs':[10, 20, 40, 60, 80],#, 120, 200, 240],
    #现在有了这个early-stopping，就可以将max_epochs设置大一点咯
    'max_epochs':[40, 60, 80, 100, 120],#, 120, 200, 240],
          
    #我之前一直很奇怪为什么会出现nan，原来学习率过大造成梯度爆炸是原因之一还有其他办法咯
    'lr': [0.0001, 0.0002, 0.0005, 0.001],#, 0.002, 0.005, 0.01, 0.02, 0.05],

    #麻痹，我看到有人说sgd的weight decay设置为0.005，我感觉把这个也当成超参搜索一下好了。
    'optimizer__weight_decay':[0, 0.001, 0.002, 0.005, 0.01],
              
    #说真的不知道怎么选择criterion就留下这两个直接进行超参搜索吧
    #我犹豫了一下这个criterion暂时不要加入超参吧，假如以后tensor的类型需要改变呢。
    #我他妈不论用中文还是英文是真的没找到所谓的选择这些超参的办法呢，每次遇到问题先手动决策criterion吧
    #'criterion':[torch.nn.L1Loss, torch.nn.SmoothL1Loss, torch.nn.MSELoss, torch.nn.BCELoss, torch.nn.CrossEntropyLoss, \
    #    torch.nn.NLLLoss, torch.nn.CosineEmbeddingLoss, torch.nn.HingeEmbeddingLoss, torch.nn.TripleMarginLoss],
    'criterion':[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
    
    'batch_size':[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    
    #'optimizer':[torch.optim.SGD(), torch.optim.RMSprop(), torch.optim.Adam()],
    #这个优化器似乎不能作为超参搜索的对象吧，因为Tutorial.py里面优化器是获得网络的参数。。
    #根据net.py中注释这个optimizer是可以设置的而且不止一种设置方式，使其可以进行超参搜索
    #尼玛，居然可以按照下面的方式进行设置的嘛，我之前为啥不行，难道是设置错了嘛？
    #下面这样的写法也不行呀，我想了一下感觉直接使用RMSprop或者是Adam吧，这两个大概率能够得到更好结果
    #毕竟我的超参搜索只能够计算有限的选择，所以我觉得选择更有可能得到更加答案的选项就行了吧，所以就是adam了
    #'optimizer':[torch.optim.SGD(net.module.parameters()), torch.optim.RMSprop, torch.optim.Adam],
    #使用Adam优化器那么下面就是准备对其关键参数进行枚举吧，意外发现adam支持L2正则化，我正准备查一下这个问题
    'optimizer__betas':[[0.86, 0.999], [0.88, 0.999], [0.90, 0.999], [0.92, 0.999], [0.94, 0.999],
        [0.90, 0.995], [0.90, 0.997], [0.90, 0.999], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999]],
        
    #这个模型初始化的部分应该在.fit()函数执行的时候吧，肯定有初始化的不然神经网络无法训练
    #其实我觉得并不需要去纠结初始化的事情，只需要知道最后搜索出来的模型的参数即可，最后发现在模型定义时初始化

    
    #下面算是模型结构参数
    #我在阅读源代码的时候终于知道下面的两个下划线连用到底是什么含义了
    #(Note that the double underscore notation in
    #``optimizer__momentum`` means that the parameter ``momentum``
    #should be set on the object ``optimizer``. This is the same
    #semantic as used by sklearn.)
    #Furthermore, this allows to change those parameters later:
    #``net.set_params(optimizer__momentum=0.99)``
    #This can be useful when you want to change certain parameters
    #using a callback, when using the net in an sklearn grid search, etc.
    #'module__num_units': [10, 20]
    
    #我觉得可以直接将这个module当做超参进行搜索就完事儿咯
    'module':[module1.weight_init1(), module1.weight_init2(), module1.weight_init3(), module1.weight_init4(), 
              module2.weight_init1(), module2.weight_init2(), module2.weight_init3(), module2.weight_init4()]
}

#突然想到一个有意思的问题，深度学习为什么需要那么多的epoch，而很多模型只是一遍学习数据
#一句简单概括，就是为了把数据中的特征进行充分学习。 那么，为什么进行一次学习学不好？
#其实，主要原因不是一次学不好，而是学不到那么好，学习容易快速的陷入局部最优，
#这样的学习容易受到初始参数，喂入数据自身特点等因素影响。为了防止快速陷入局部最优，
#大家最常用的方法就是使用较小的学习速率，也就是降低learning rate。learning rate小了，
#为了将数据学充分，自然就要增加学习次数。也就是增加epoch.作者：Ryan，来自知乎，妈的那个解惑者学院的创始人之一吧？

#我理解深度学习如果学习率设置过大很容易梯度爆炸，整个神经网络就废了（效果很差）。
#那么为了避免梯度爆炸往往将学习率设置的很小，为将数据学充分自然需要增加学习次数。

#我起先以为test_size的设置越小越好，但是似乎不是这个样子的，我只能说隐约觉得和过拟合有关系
#搞清楚了超参搜索其实是计算了cv*n_iter次数咯，然后我觉得可以设置test_siez=0.1这样用了81%的例子进行模型学习
#我用了81%的数据进行学习，感觉这样才取得了比较靠谱的结果呢。搞清楚原理才是真的，不然很鸡巴容易搞成玄学。
#过拟合的原因在于（1）模型过于复杂。（2）模型迭代的次数过多。 卧槽逻辑回归也可以学习多个epochs？？？炫酷
#如果不是用于测试比较适合的噪声，那么test_size设置为0.01其实最好的，我应该关注score1与score2而不是score3!
X_split_train, X_split_validate, Y_split_train, Y_split_validate = \
    train_test_split(X_train_scaled, Y_train, test_size =0.01, shuffle=True)

#我还以为是cv=3造成的比较好的结果，原来设置cv=10结果也不错的呀，神经网络远强于传统模型。。
#其实搞清楚了（随机）超参搜索的原理，以及本持者更多的数据大概率更好的结果,n_folds改为20.。
skf = StratifiedKFold(Y_split_train, n_folds=20, shuffle=True, random_state=None)

#refit默认是True，直接使用默认值就好了，不用在RS设置refit=True。反倒是scoring='accuracy'可以设置一下
#因为scoring的默认值是none。如果设置为None的时候模型的默认scorer或者说metrics会被调用
#因为skorch版本的模型并没有默认的socring设置所以超参搜索的时候必须手动设置否则无法运行咯
#那么又有一个问题咯，就是net.score他妈的用的到底是啥准则呢？这个会不会是refit设置为False的理由呢？
#这个应该不是refit设置为False的理由吧，如果net.score没有准则或者无法设置准则的话超参搜索也无法设置准则的吧？
#这个n_iter对于其他模型的时候似乎是控制迭代次数，但是对于NeuralNetClassifier好像是计算了cv*n_iter次数咯。
#经过我的验证device设置为cuda确实比cpu快太多了，而且对于NeuralNetClassifier确实是计算了cv*n_iter次数咯。
#他妈的，可算是被我测出来了，这个随机超参搜索用同一个net计算cv然后选择最好的那个。。可能sklearn的超参就这么搞的，我才知道。。

"""
#我还以为是net的fit函数存在一定的问题，这个实验说明问题不在这里吧
net.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong))
Y_pred = net.predict(X_split_train.values.astype(np.float32))
counts = (Y_pred==Y_split_train).sum()
print("准确率为：",counts/len(Y_split_train)) 
"""

#感觉除了层数和每层隐节点的个数，也没啥好调的。其它参数，近两年论文基本都用同样的参数设定：
#迭代几十到几百epoch。sgd，mini batch size从几十到几百皆可。步长0.1，可手动收缩，weight decay取0.005，
#momentum取0.9。dropout加relu。weight用高斯分布初始化，bias全初始化为0。最后记得输入特征和预测目标都做好归一化。
#做完这些你的神经网络就应该跑出基本靠谱的结果，否则反省一下自己的人品
search = RandomizedSearchCV(net, params, cv=skf, n_iter=1, scoring='accuracy')
search.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong))
score1 = search.best_score_
score2 = search.score(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.longlong))
score3 = search.score(X_split_validate.values.astype(np.float32), Y_split_validate.values.astype(np.longlong))

#我之前突然再想这个score1、score2、score3再想他是咋评价出来的
#然后楞了一下才想起这个RandomizedSearchCV机制是选出交叉验证结果最好的
#这个其实都是小问题咯，真正稍微大一点的问题可能是max-epoch
#之前的sklearn基本都是看是否收敛决定迭代次数且不超过max-epoch
#就目前而言，我发现skorch似乎是必须计算到max-epoch才会停止的
#我觉得更麻烦的问题可能是这个scoring的问题，他涉及到了默认scorer或者metrics
#通过调试代码可以发现我之前使用的逻辑回归等模型默认的scoring选择正是'accuracy'.
#2018-7-24 10:51:53 
#那么接下来的事情是准备考虑一下max_epochs、网络结构、opt以及cri的问题咯
#我查了一下sklearn中的max_iter，其决定了最大epoch或者tol决定的收敛程度
#我觉得skorch中的max_epochs应该也实现max_iter这种才比较人性化吧
#mlp = MLPClassifier()
#我查了包含NeuralNetClassifier的net.py并没有类似max_iter或者tol的属性
#而且我运行了print(net.get_params().keys())输出了net的所有属性确实没有。。
#2018-7-24 11:19:40
#接下来就试一下这个opt与cri如何设置呢，主要是opt还需要进一步设置的吧？
#现在看来opt只有都设置好了之后放到list中进行筛选咯，在外面无法组合搜索了

#（4）实现battle的框架三咯，最好加入蚁群算法或者遗传算法，但是模型之间如何比较？
#（5）然后是特征工程的框架四（超远景吧），或许以后比拼的就是这个东西了吧

#Usually, the gradients become NaN first. The first two things to look 
#at are a reduced learning rate and possibly gradient clipping.
print(score1)
print(score2)
print(score3)

"""
columns = [3, 4, 5, 6, 7, 8]
mean = [0, 0, 0, 0, 0, 0, 0]
sigma =[0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.11, 0.13, 0.15, 0.17, 0.20]
#下面这个函数确实是不能够直接调用score函数
#然后我也不可能针对skorch或者pytorch实现score吧
#那就修改一下skorch或者pytorch版本的数据增强函数吧
#迄今为止，我想到的最低成本的办法就是利用现成的就行了吧吧
#最关键的问题在于不实现这个函数不好比较这些模型之间的区别呢，我想了一下感觉还是只有实现这个函数咯
#进行过这个实验之后才感觉有点绝望了吧，一晚上仅仅写入了一次times的结果（70次计算）连140次都没有
#按照这个情况如果使用遗传算法应该更是无解的慢吧，毕竟50个样本一代就要计算500次的意思咯。今晚上最后一次计算然后提交了吧
noise_augment_pytorch_classifier(10, 10, X_train_scaled, Y_train, 0.1, net, params, mean, sigma, columns, 900, n_iter=10)
"""