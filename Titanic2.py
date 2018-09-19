#coding=utf-8
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

#以后只有非常重要的东西才两边都传递吧，不然一直单边存在文件就行了。
#根据二八定律这种文件存在的极少所以不用特别担心呢。以后文件不再传递Titanic吧。

#这个原来版本的代码怎么调试都有点问题，算了感觉还是自己写一个吧，比调试学到的东西多
X_train_scaled_array = X_train_scaled.values
Y_train_array = Y_train.values
x_train, x_val, y_train, y_val = train_test_split(X_train_scaled_array, Y_train_array, test_size = 0.1)
"""
x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(X_train_scaled, Y_train, test_size = 0.1)
x_train = x_train_df.values
x_val = x_val_df.values
y_train = y_train_df.values
y_val = y_val_df.values
"""

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 270)
        self.fc2 = nn.Linear(270, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        #输出层好像一般都是用sigmoid函数咯Titanic2也是用的sigmod
        #也可以不用sigmoid函数吧，只是需要解决预测结果问题咯
        x = F.sigmoid(x)
        #如果将上面的sigmoid函数修改为relu函数其他代码基本通用的
        #但是修改成这个函数以后好像结果不是很稳定高的时候很高低的时候很低
        #我印象中好像推荐一般输出层采用sigmoid函数吧，这个问题就这么招了
        #x = F.relu(x)
        
        return x
    
net = Net()
#我将num_epochs和learning_rate分别修改为80和0.001然而并没有啥作用
batch_size = 50
num_epochs = 50
learning_rate = 0.01
#我在修改Titanic1代码的时候遇到了两个问题，其中一个问题是
#老子调试了这么久发现原来问题出在这里？？这个batch_no其实是做了一个除法的？？
#只不过这个写法感觉非常的非主流当时没多想，我是说怎么看也会越界的吧？？
#另外一个问题是不能传入dataframe只能够传入numpy.ndarray感觉非常不健壮
#下面这行代码居然是做除法然后取整，我是真的没看懂这么飘逸的写法
#batch_no = len(X_train)//batch_size #batch_size
#我个人比较熟悉的其实是下面的这种写法呀
#batch_no = int(len(X_train)/batch_size) 
#上面两种写法代码都有点问题：将测试集的数据用于训练了，算是原作者留下的一个坑
#卧槽，上面的写法并没有将测试集用于训练，只不过这个写法会让test_size修改出异常
#我现在将原作者代码修改如下，并将test_size修改为0.3，这样才能准确比较模型优劣吧
#就以上这些实验条件的限制下，目前神经网络的结果还不如xgboost的吧？然后我将参数改回去
#我在想如何将xgboost等模型的test_size改回去应该准确率也有不错的提升吧
#所以感觉这个test_size参数其实也是一个超参的感觉，0.3也算是经验参数吧，估计0.1也算吧
batch_no = int(len(x_train)/batch_size) 
#通过调试我发现我对Pytorch的很多细节并不了解，所以决定写一下注释
#这个其实是个损失函数吧，就是交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#这个优化器其实就是训练方式，SGD之类的其实都是训练方式
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#这个epoch决定训练集训练多少遍
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch+1))
    #这个shuffle其实是将训练集打乱的意思肯定要加入一些随机元素的
    #仔细看这个划分batch的方式，并不是epoch所有数据都被用于训练
    #造成这个情况本质上是batch的大小。只有训练集的样本总数目整除batch才行
    x_train, y_train = shuffle(x_train, y_train)
    # Mini batch learning
    #这个代码是有问题的吧？i in range会直接越界的吧？所以问题就在这里。。
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        #为什么在笔记本上面执行到这里就会卡住？难道是CPU版本Pytorch的锅？
        #笔记本上面运行到这里就卡住了但是并没有报错，GPU上面报错但是看不懂呢
        #但是Titanic3.py里面都是可以正常执行的，这到底又是什么原因呢？
        #让我觉得很困惑的原因在于，这个为什么不报错呢？真的哔了狗了？？？
        #我查了很久原来Titanic3里面是numpy.ndarry然后Titanic1是里面是Dataframe
        #果然采用numpy.ndarray转换之后，在将X_train改为X_train_scale下面一行不报错了
        #所以Pytorch的库感觉完全没有sklearn的健壮吧，sklearn传入ndarry或者df几乎没影响
        x_var = Variable(torch.FloatTensor(x_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))
        # Forward + Backward + Optimize
        #zero_grad()清空所有被优化过的Variable的梯度。
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss =criterion(ypred_var, y_var)
        #这个应该是反向传播误差吧？
        loss.backward()
        #进行单次优化 (参数更新)。
        optimizer.step()
#我觉得Pytorch的错误提示真心糟糕，很像找机会把她亲妈杀了
#我对这个代码的水平感觉比较质疑，这尼玛的写的是啥呀？
test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
with torch.no_grad():
    #result返回的结果是死亡的概率和存活的概率，这个有点意思咯
    result = net(test_var)
#result、values以及labels都是tensor类型的变量
#我感觉下面几行代码很迷惑的样子呢，尤其是max函数
#可能之所以会出现下面的情况就是因为使用了sigmoid激活函数？
#torch.max()返回两个结果，第一个是最大值，第二个是对应的索引值；
#第二个参数 0 代表按列取最大值并返回对应的行索引值，1 代表按行取最大值并返回对应的列索引值。
#神经网络只是在拟合输入与输出，真正需要思考一下的其实是max函数
#为什么使用max函数而不是用min函数是因为max结果刚好符合预期
#r如果在原问题中survived字段中0与1含义相反，那么使用min函数
values, labels = torch.max(result, 1)
#下面反馈的结果好像有点意思的感觉呢
#>>> type(labels)
#<class 'torch.Tensor'>
#>>> type(labels.data)
#<class 'torch.Tensor'>
#>>> type(labels.data.numpy)
#<class 'builtin_function_or_method'>
#>>> type(labels.data.numpy())
#<class 'numpy.ndarray'>
num_right = np.sum(labels.data.numpy() == y_val)
print('Accuracy {:.2f}'.format(num_right / len(y_val)))
#我真的日了尼玛了，感觉用了七八个小时才解决这个小问题。。
