#coding=utf-8
import os
import sys
import random
import pickle
import datetime
import numpy as np
import pandas as pd
sys.path.append("D:\\Workspace\\Titanic")
from Utilities1 import noise_augment_pytorch_classifier

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import collections
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import skorch
from skorch import NeuralNetClassifier

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc

def print_nnclf_acc(acc):
    
    print("the accuracy rate of the model on the whole train dataset is:", acc)
    
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


"""
#这个写法因为OrderedDict必须是字典才行，所以不OK咯
def auto_module_creator(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    layers_list=[]
    
    if (hidden_layers==0):
        layers_list.append(nn.Linear(input_nodes, output_nodes))
        #我觉得连接输出层的就不需要进行dropout了吧，可能影响效果呢
        #layers_list.append(nn.Dropout(percentage))
        
    else :
        layers_list.append(nn.Linear(input_nodes, hidden_nodes))
        layers_list.append(nn.Dropout(percentage))
        for i in range(0, hidden_layers):
            layers_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers_list.append(nn.Dropout(percentage)) 
        layers_list.append(nn.Linear(hidden_nodes, output_nodes))
    
    return nn.Sequential(collections.OrderedDict(layers_list))
"""

"""
#这个写法报错说F.relu()输入的必须是tensor而不是nn.linear之类的东西
def auto_module_creator(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list=[]
    
    if (hidden_layers==0):
        
        #这种写法理论上好像没有问题了吧，之前一直搞不定softmax的部分咯
        module_list.append(F.softmax(F.relu(nn.Linear(input_nodes, output_nodes)), dim=-1))
        
    else :
        module_list.append(F.relu(nn.Linear(input_nodes, hidden_nodes)))
        module_list.append(nn.Dropout(percentage))
        for i in range(0, hidden_layers):
            module_list.append(F.relu(nn.Linear(hidden_nodes, hidden_nodes)))
            module_list.append(nn.Dropout(percentage)) 
        module_list.append(F.softmax(F.relu(nn.Linear(hidden_nodes, output_nodes), dim=-1)))
    
    #那就只有通过下面的方式避免取名字咯
    model = nn.Sequential()
    for i in range(0, len(module_list)):
        model.add_module(str(i+1), module_list[i])
    
    return model
"""

"""
#之前未将F.relu()和F.softmax()替换为nn.ReLU()和nn.Softmax()输出如下错误
#TypeError: relu() missing 1 required positional argument: 'input'
#其实下面的这个写法可以用了，但是我还是想要更加优雅一点的写法
def auto_module_creator(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    m=0
    model = nn.Sequential()
    
    if(hidden_layers==0):
        
        m+=1
        model.add_module(str(m), nn.Linear(input_nodes, output_nodes))
        m+=1
        model.add_module(str(m), nn.ReLU())
        m+=1
        model.add_module(str(m), nn.Softmax())
        
    else :
        m+=1
        model.add_module(str(m), nn.Linear(input_nodes, hidden_nodes))
        m+=1
        model.add_module(str(m), nn.ReLU())
        m+=1
        model.add_module(str(m), nn.Dropout(percentage))
        
        for i in range(0, hidden_layers):
            m+=1
            model.add_module(str(m), nn.Linear(hidden_nodes, hidden_nodes))
            m+=1
            model.add_module(str(m), nn.ReLU())
            m+=1
            model.add_module(str(m), nn.Dropout(percentage))
             
        m+=1     
        model.add_module(str(m), nn.Linear(hidden_nodes, output_nodes))
        m+=1
        model.add_module(str(m), nn.ReLU())
        m+=1
        model.add_module(str(m), nn.Softmax())
    
    return model
"""

#我的天，我刚才发愁写不出这个东西然后去床上睡了四个小时
#在睡觉的时候做了很多的梦得到了很多的神级启示惊呆了我
#比如说我梦到神经网络的一百个证明，证明了98个利用了两个公理
#我不知道是不是已经存在神经网络的一百个证明，还是神启发我去做这事儿
#如果想要做这件事情可能需要torch.nn.Sequential的帮助吧，不然感觉挺麻烦的
#Andrew Ng在ML课程中提到输入层：神经元个数=feature维度
#输出层：神经元个数=分类类别数。隐层： 默认只用一个隐层
#如果用多个隐层，则每个隐层的神经元数目都一样
#隐层神经元个数越多，分类效果越好，但计算量会增大
#我感觉在设计之前是不是在过一遍他们的之前的讲义呢？
#神经网络的普适性原理很炫酷：可以用一层隐函数拟合任何函数（足够的隐藏单元）
#李宏毅的thin all vs fat short.这个东西真的可以自动设计吗，那么多元素在内需要考虑
#nn.Conv1d 2d 3d,nn.Linear,F.relu,F.max_pool2d,
#https://ptorch.com/docs/1/torch-nn这里面其实有所有的nn相关的元素
#看了这个以后我觉得只能够使用nn.Sequential不然我不可能找出新的规范的吧
#我通过ctrl代码中Sequential的方式查到类似的还有ModuleList和ParameterList
#所以我现在似乎找到了思路，下面这个函数生成的东西经过ModuleList之后变为模型咯
#我之所以迟迟无法下手的缘故就是因为想要找到能够面面顾及的解决方案，但是并不存在吧
#这个问题的难点主要在：1）接口设计，兼容我的超参甚至是其他超参 2）模型的保存问题咯
#以下是今天的任务咯：
#1）想清楚超参搜索的次数以及结构是否符合要求？如果真的是3000次还没达到最优的话
#那以后在更大的数据集上面还怎么选择超参，你他妈真的在逗我吧。。所以这次一定要
#确定贝叶斯优化的次数以备以后使用咯。超大规模的数据你在说尼玛呢。。只有xgboost吧
#A）或者其实贝叶斯的优化次数依据具体计算时间咯，如果时间比较有限那就直接使用经验参数
#最少还是需要1000次或者一天计算量中的最多者吧，太少了还是没啥意义的咯根本不准。
#或者依据减少超参的数目，多提供经验超参进行贝叶斯优化的时候或许能够取得更佳效果。
#B）这么大规模的数据，性能的提升根本没办法依靠超参搜素吧，换成xgboost或许还能玩一下超参。
#但是你使用统计模型太依赖数据的特征了吧，反复修改特征在超参一下xgboost还是很费时间的
#C）综上所述，看短期的话还是神经网络最简单无脑，看长期的话神经网络肯定是未来的趋势咯
#神经网络实在不行就用经验参数吧，又能少做特征工程也能自动调参，模型提升主要靠G、S、A等。
#上述的G代表对抗网络、S代表Stacking、A代表模型融合咯。说实话超参搜索的红利就这么多了吧。。
#所以这个结构选择的版本就不修改带the_end_of_the_titanic咯，titanic的版本是不带结构搜索的。。
#2）完成代码的再优化咯
#3）GAN、S、A等
#4）完成下一轮的超参搜索
#假设输入参数均正常咯，我自己使用的时候就不考虑这些奇葩因素咯
#卷积池化那些操作过于复杂已经没有办法去做了，将就这样的吧
class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X
    
#有点绝望了耶，好像只能够使用modulelist解决这个问题吧。
#因为nn.sequential的两种方式都不是可以直接传入List的
#此外还有一些很奇怪的事情，比如说什么F.relu缺少参数。。
#昨天要下班的时候终于查到了modulelist并没有实现forworad函数
#怪不得之前使用modulelist都没有成功，还好查到了sequential的第三种用法
#说实在的，就算这个算法写好了我还是有点担心这些超参增加计算的时间呢
#或许我现在是真的不得不用GPU来学习神经网络了吧，但是有一点是肯定的
#通过网络结构来获得更好的准确率带来的结果提升差不多就是这么多了吧。。
#下面的这个是第一个版本的auto_module_creator1咯，我觉得为了易用性必须还有其他版本吧。
#这个版本是每个nn.Linear后面都必须带上dropout的版本感觉不够稳定的吧，所以有了auto_module_creator2咯
def auto_module_creator1(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list = []
    
    if(hidden_layers==0):
        
        module_list.append(nn.Linear(input_nodes, output_nodes))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
        
    else :
        module_list.append(nn.Linear(input_nodes, hidden_nodes))
        module_list.append(nn.ReLU())
        module_list.append(nn.Dropout(percentage))
        
        for i in range(0, hidden_layers):
            module_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(percentage))
             
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
    
    model = nn.Sequential()
    for i in range(0, len(module_list)):
        model.add_module(str(i+1), module_list[i])
        
    return model

#我这个版本还是修改为三行插入一个dropout吧
#至于以后的不同的网络结构的话，再设计不同的结构吧
#其实理论上进化算法似乎可以进行网络结构的选择，然后选出结构在进行超参搜索吗？
#我觉得这个想法算是一种解决问题的思路咯，下个目标就是准备实现这个进化算法咯
#不得不说，这样子设计模型感觉还是比之前设计模型效率高一些的吧
def auto_module_creator2(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
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

#我好想找到问题所在了吧，只是建立了模型并没有forword所以无法训练咯
#看来解决方案还是要采用nn.Sequential才是简单的实现方案呢
#nn.Modulelist还需要自己实现forword之类的方案，这就是之前报错的原因
#这个dropout的设置值似乎很需要经验和运气的样子，随便设置可能都没卵用的
#不使用隐藏层感觉弱爆了的样子呢，收敛更慢一些，大体性能上没有一层隐藏层60节点好用
for i in range(0, 10):
    model = auto_module_creator2(9, i, 35, 2, 0.00)
    print(model)
    print()

clf = NeuralNetClassifier(lr = 0.0005,
                          optimizer__weight_decay = 0.001,
                          criterion = torch.nn.CrossEntropyLoss,
                          batch_size = 128,
                          optimizer__betas = [0.90, 0.9999],
                          module=model,
                          max_epochs = 400,
                          callbacks = [skorch.callbacks.EarlyStopping(patience=5)],
                          device = "cpu",
                          optimizer = torch.optim.Adam
                          )
clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 

metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
print_nnclf_acc(metric)
        
files = open("test_best_model.pickle", "wb")
pickle.dump(clf, files)
files.close()

files = open("test_best_model.pickle", "rb")
clf = pickle.load(files)
files.close()

metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
print_nnclf_acc(metric)