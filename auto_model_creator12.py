#coding=utf-8
#这个版本的目的在于使用自助法来训练单模型进而使stacking的神经网络差异足够大
#最后仅仅是添加了一些噪声而已，并没有使用自助法仅仅是通过添加噪声的方式去加大了模型之间的差异
#但是加了噪声的版本也就是追平了之前的记录而已呀，然后我试了单模型多次搜索并交叉验证但是结果只能说还行。
#我之前还想使用删除异常点的办法做最后一次实验，但是觉得删除异常点和增加噪声应该差不多的吧
#所以最后的实验是stacking加上增加噪声，但是多增加几次对于增加过噪声之后的数据的搜索吧。
#或者试一下单节点加上噪声泛化效果如何呢？我觉得单节点怎么增加噪声都比不上多节点增加噪声吧。。
#最后整理一下思路，单节点肯定不上stacking，然后我目前的问题是单模型如何优化，以及如何增加模型的多样性
#单模型增强的话之前尝试了更高的折数的交叉验证提升比较有限，现在只有用噪声咯，这大概就是最后一次试验了吧。

#修改内容集被整理如下：
#（0）到这个时候我才发现GPU训练神经网络的速度比CPU训练速度快很多耶。不对呀，好像也没有快很多吧
#现在看来可能是和昨天CPU在运行别的程序有关吧导致计算比较慢，GPU似乎并没有比CPU带来十倍的优势吧？
#所以我觉得可能是我买的台式机被人给坑了吧，不过好在还有GPU可用。就是每次运行之前需要设置device和path咯。
#（1）将保存文件的路径修改了。
#（2）特征处理的流程需要修改。尤其是可能增加清除离群点的过程。
#（3）record_best_model_acc的方式可能需要修改，或许我们需要换种方式获取最佳模型咯，不对好像暂时还不能修改这个东西。
#（4）create_nn_module函数可能需要修改，因为每层都有dropout或者修改为其他结构如回归问题咯。
#（5）noise_augment_dataframe_data可能需要修改，因为Y_train或许也需要增加噪声的。
#（6）nn_f可能需要修改，因为noise_augment_dataframe_data的columns需要修改咯，还有评价准则可能需要优化或者不需要加噪声吧？但是暂时不知如何优化
#（7）nn_stacking_f应该是被弃用了，因为之前我尝试过第二层使用神经网络或者tpot结果都不尽如人意咯，第二层使用逻辑回归才是王道。
#（8）parse_nodes、parse_trials、space、space_nodes需要根据每次的数据修改，best_nodes本身不需要主要是为了快速测试而存在。
#（9）train_nn_model、train_nn_model_validate1或许需要换种方式获取最佳模型咯。现在已经找到最佳方式选择模型咯
#（10）nn_stacking_predict应该是被弃用了，因为这个函数是为单模型（节点）开发的预测函数。
#（11）lr_stacking_predict应该是被弃用了，因为这个函数没有超参搜索出最佳的逻辑回归值，计算2000次结果都是一样的。
#（12）tpot_stacking_predict应该是被弃用了，因为第二层使用神经网络或者tpot结果都不尽如人意咯，第二层使用逻辑回归才是王道。
#（13）get_oof回归问题可能需要改写
#（14）train_nn_model、train_nn_model_validate1、train_nn_model_noise_validate2这三系列函数可能需要修改device设置和噪声相关设置。
#模型的整体提升划分为四块：从数据上提升性能、从算法上提升性能、从算法调优上提升性能、从模型融合上提升性能（性能提升的力度按上表的顺序从上到下依次递减。）
#具体内容可参加https://www.baidu.com/link?url=zdq_sTzndnIZrJL71ZFaLlHnfSblGnNXPzeilgVTaKG2RJEHTWHZHTzVkkipM0El&wd=&eqid=aa03b37b0004b870000000025c2f02e6
#（15）可能以后就是增加正则化项吧，能够一定程度的减小网络的复杂度类似奥卡姆剃刀原则。自己随机生成大量的数据吧。将数据缩放到激活函数的阈值内
#原来神经网络模型的训练一直就比较慢，以至于有的时候不一定要采用交叉验证的方式来训练，可能直接用部分未训练数据作为验证集。。
#然后对于模型过拟合或者欠拟合的判断贯穿整个机器学习的过程当中，原来stacking其实是最后一种用于提升模型泛化性能的方式咯。我的面试可以围绕这些开始吧。

#我到今天才知道dataframe是一列一列的而ndarray是一行一行的？？不过之前的函数测试都是木有问题的哈，这就很好咯
import os
import sys
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

#原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction import DictVectorizer

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
from sklearn.linear_model.logistic import LogisticRegressionCV
from nltk.classify.svm import SvmClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
#下面的这个kfold是实现k折交叉的功能，返回每次的indice，可以设置为shuffle但默认未设
#然后这个StratifiedKFold是返回k折交叉的迭代器，每次通过迭代器返回结果，可以设置为shuffle
#两者的区别在于前者返回indice或者索引列表后者直接返回迭代器，虽然我这一份代码两种方式都有但是让他们并存吧
#from sklearn.model_selection import KFold,StratifiedKFold

warnings.filterwarnings('ignore')

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

data_train = pd.read_csv("C:/Users/1/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/1/Desktop/test.csv")
combine = [data_train, data_test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Capt'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
"""
    DictVectorizer类只能够映射非数值的类，所以取消了这里的映射
    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)
"""

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

#现在这里的性别不能够被替换否则下面的
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
    
#为了之后使用dictvector进行映射，现在又将性别变为字符串
#如果不用这样的操作变为字符串就无法用dictvector替换啦
#因为dictvector的替换就是直接将非数值的东西变成单独的一类
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({1:'female', 0:'male'})

"""
#我觉得年龄好像没有必要划分的吧，就删除下面这段也挺好的
#我在想这样划分了年龄之后是不是限制了模型的拟合过程
#毕竟神经网络理论上是能够以任何精度拟合任何函数的吧？
#我个人的理解这样子划分之后损失了原来的信息，导致函数性能无法提高咯。。
for dataset in combine: 
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0 
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1 
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2 
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3 
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
"""   
    
#这里的mode是求解pandas.core.series.Series众数的第一个值（可能有多个众数）
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
"""
DictVectorizer类只能够映射非数值的类，所以取消了这里的映射
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 
"""

#将data_test中的fare元素所缺失的部分由已经包含的数据的中位数决定哈
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

"""
我觉得这个船票的费用好像也不用这样子处理吧，试一下原生态的感觉呢
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
"""

#按照下面的写法那么Cabin肯定都是1了呀，所以我自己写出了下面的写法咯
#for dataset in combine:
#    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = 0
#    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = 1
for dataset in combine:
    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = "cabin"  
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = "no cabin" 
    
#为了强行把Pclass进行One-Hot编码强行进行下面的转换。。
#当执行到下面的for循环的时候可以查看下面的数据
#下面的反馈显示data_train和combine[0]对应于同一块存储空间
#>>> id(data_train)
#2789318948624
#>>> id(data_test)
#2789319071560
#>>> id(combine[0])
#2789318948624
#>>> id(combine[1])
#2789319071560
for dataset in combine:
    #dataset['Pclass'] = dataset['Pclass'].map({1: "1st", 2: "2nd", 3: "3rd"})
    dataset.loc[ dataset['Pclass'] == 1, 'Pclass'] = "1st"
    dataset.loc[ dataset['Pclass'] == 2, 'Pclass'] = "2nd"
    dataset.loc[ dataset['Pclass'] == 3, 'Pclass'] = "3rd"

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
        #主要是为了使用DictVectorizer类映射所以改写下面的样子
        #ticket = 1
        ticket = "share"
    else:
        #ticket = 0                   #遍历所有船票，在共享船票里面的为1，否则为0
        ticket = "no share"
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
        #ticket = 1
        ticket = "share"
    else:
        #ticket = 0                  
        ticket = "no share"
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_test = pd.concat([data_test, results], axis=1) 

"""
#为了强行把Pclass进行One-Hot编码强行进行下面的转换。。
#执行到下面的for循环内部的时候可以查看下面的数值。
#可以发现上面的pd.concat导致data_train和data_test id改变
#所以我就纳闷了，下面的for循环代码为什么没有修改到文件中的数据
#原来是因为pd.concat创建了新的对象，下面的代码改的是原来的数据。。
#OK,花了一个小时的时间才搞清楚问题。。现在可以把下面的代码注释掉了吧
#>>> id(data_train)
#2789320495400
#>>> id(data_test)
#2789210849408
#>>> id(combine[0])
#2789318948624
#>>> id(combine[1])
#2789319071560
for dataset in combine:
    #dataset['Pclass'] = dataset['Pclass'].map({1: "1st", 2: "2nd", 3: "3rd"})
    dataset.loc[ dataset['Pclass'] == 1, 'Pclass'] = "1st"
    dataset.loc[ dataset['Pclass'] == 2, 'Pclass'] = "2nd"
    dataset.loc[ dataset['Pclass'] == 3, 'Pclass'] = "3rd"
"""

data_train_1 = data_train.copy()
data_test_1  = data_test.copy()
data_test_1 = data_test_1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1)

X_train = data_train_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]
Y_train = data_train_1['Survived']

X_test = data_test_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]

X_all = pd.concat([X_train, X_test], axis=0)
#print(X_all.columns)
#下面是我补充的将性别、姓名、Embarked修改为了one-hot编码类型了
#原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
dict_vector = DictVectorizer(sparse=False)
X_all = dict_vector.fit_transform(X_all.to_dict(orient='record'))
X_all = pd.DataFrame(data=X_all, columns=dict_vector.feature_names_)
#print(X_all.columns)到这里已经是ndarray已经没有了columns咯
#print(dict_vector.feature_names_)

#这个主要是为了测试写出来的文件是正确的。
#output = pd.DataFrame(data = X_all)            
#output.to_csv("C:/Users/1/Desktop/dict.csv", columns=X_all.columns, index=False)
            
#我觉得训练集和测试集需要在一起进行特征缩放，所以注释掉了原来的X_train的特征缩放咯
#用了五个月之后我发现我的特征缩放好像做错了？？所以试一下下面的特征缩放吧。。不过变量名好像可以不用修改吧
#原来这两种特征缩放的结果是差不多的，我勒个去当时吓我一跳我我还以为这么久的工作都白做了呢。
#X_all_scaled = pd.DataFrame(preprocessing.scale(X_all), columns = X_train.columns)
#X_all_scaled = pd.DataFrame(StandardScaler().fit_transform(X_all), columns = X_train.columns)
X_all_scaled = pd.DataFrame(StandardScaler().fit_transform(X_all), columns = X_all.columns)
#X_train_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)
#https://blog.csdn.net/CherDW/article/details/56011531讲解了几种特征缩放的区别，scale和.StandardScaler其实差不多。。
X_train_scaled = X_all_scaled[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]

#这个主要是为了测试特征缩放之后的结果是正常的
#下面特征缩放之后的结果看起来很壮观的样子23333。
#output = pd.DataFrame(data = X_all_scaled)            
#output.to_csv("C:/Users/1/Desktop/dict_scaled.csv", columns=X_all.columns, index=False)

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

#下面这个方式修改代码是最简单对于全局影响最小的方式了吧
#可能每次得到的stacked_train不一样所以保存的best_model并没有那么有意义
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

#经过这么长时间的了解，我觉得还是应该每层都加上dropout比较好一些的
def create_nn_module(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list = []
    
    #当没有隐藏节点的时候
    if(hidden_layers==0):
        module_list.append(nn.Linear(input_nodes, output_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
        
    #当存在隐藏节点的时候
    else :
        module_list.append(nn.Linear(input_nodes, hidden_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        
        for i in range(0, hidden_layers):
            module_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            module_list.append(nn.Dropout(percentage))
            module_list.append(nn.ReLU())
             
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        module_list.append(nn.Softmax())
            
    model = nn.Sequential()
    for i in range(0, len(module_list)):
        model.add_module(str(i+1), module_list[i])
    
    return model
"""
def create_nn_module(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
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

"""
model1 = create_nn_module(3, 0, 3, 3, 0.1)
print(model1)
model2 = create_nn_module(3, 3, 3, 3, 0.2)
print(model2)
model3 = create_nn_module(3, 4, 4, 2, 0.2)
print(model3)
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
        
def noise_augment_dataframe_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] +=  random.gauss(mean, std)

    return X_noise_train, Y_train

def noise_augment_ndarray_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train[i][j] +=  random.gauss(mean, std)
    
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
        
    X_noise_train, Y_noise_train = noise_augment_dataframe_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[i for i in range(1, 20)])#columns=[])
    
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              module = create_nn_module(params["input_nodes"], params["hidden_layers"], 
                                                      params["hidden_nodes"], params["output_nodes"], params["percentage"]),
                              max_epochs = params["max_epochs"],
                              callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                              device = params["device"],
                              optimizer = params["optimizer"]
                              )
    #这里似乎可以采用分层采样吧，原来这个就是分层随机采样，妈的吓我一跳还以为要重新修改。
    #我现在在train_test_split中也采用了分层划分的数据集，一般使用分层的效果更好一点的。
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    init_module(clf.module, params["weight_mode"], params["bias"])
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric

#这个函数已经弃用了，因为第二层使用逻辑回归是目前最好的选择
#之前我尝试过第二层使用神经网络或者tpot结果都不尽如人意咯。
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
    #X_noise_train, Y_noise_train = noise_augment_dataframe_data(params["mean"], params["std"], stacked_train, Y_train, columns=[i for i in range(0, stacked_train.columns.size)])
    X_noise_train, Y_noise_train = noise_augment_dataframe_data(params["mean"], params["std"], stacked_train, Y_train, columns=[])
    
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              #为了不再重新创建space,space_nodes就用下面的写法吧
                              module = create_nn_module(stacked_train.columns.size, params["hidden_layers"], 
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

#这个选择最佳模型的时候存在过拟合的风险
def train_nn_model(nodes, X_train_scaled, Y_train, max_evals=10):
    
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
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
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

#我尽量用了一点别的方式减小模型选择时候可能带来的过拟合风险吧
#为了不改变原来参数的接口或者以最小修改代价的方式修改代码我想到了下面的办法咯
#下面的修改方式比我之前想到的修改方式感觉上还要高明一些的呢。。
#下面的修改方式确实是比较高明呀，在get_oof_validate1每次从训练集中抽取0.05
#而不是从整体中抽取0.05，一来确实能够增加模型之间的多样性，二来更充分利用了数据吧。
def train_nn_model_validate1(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #我觉得0.12的设置有点多了，还有很多数据没用到呢，感觉这样子设置应该会好一些的吧？
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.05, stratify=Y_train)
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
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
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

#这里面采用cross_val_score的方式应该更能够体现出泛化的性能吧。
#这样的交叉验证才是最高效率的利用数据的方式吧。
def train_nn_model_validate2(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #解决这个问题主要还是要靠cross_val_score这样才能够显示泛化性能吧。
    best_acc = 0.0
    best_model = 0.0
    for j in range(0, max_evals):
        
        clf = NeuralNetClassifier(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(clf.module, nodes["weight_mode"], nodes["bias"])
        
        #这边的折数由5折修改为10折吧，这样子的话应该更加能够表示出稳定性吧
        skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)
        metric = cross_val_score(clf, X_train_scaled.astype(np.float32), Y_train.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
        #这里测试一下如此修改能够达到目的呢，这样的方式应该比之前靠谱多了吧，经过测试
        #我觉得cross_val_score确实更可以表示泛化能力，验证设置为10比5总体而言更准确
        #clf.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
        #score = cal_nnclf_acc(clf, X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
        #print(metric)
        #print(score)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
    return best_model, best_acc

def train_nn_model_noise_validate1(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #我觉得0.12的设置有点多了，还有很多数据没用到呢，感觉这样子设置应该会好一些的吧？
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.05, stratify=Y_train)
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
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
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

#然后在这里增加一次噪声和验证咯，感觉我把程序弄的真的好复杂呀？
#或许我下一阶段的实验就是查看是否nn_f不加入噪声只是第二阶段增加噪声效果是否更好？
def train_nn_model_noise_validate2(nodes, X_train_scaled, Y_train, max_evals=10):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.1, stratify=Y_train)

    best_acc = 0.0
    best_model = 0.0
    for j in range(0, max_evals):
        
        clf = NeuralNetClassifier(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(clf.module, nodes["weight_mode"], nodes["bias"])
        
        #这里需要实现噪声的增加以防止模型过拟合
        #下面的两行代码的写法会导致竖着写，应该需要按照没有被注释掉的方式写。
        #下面是来行的输出显示两种写法好像是等价的，我的天浪费了这么多时间研究这个。。最后居然是一样的
        #我刚才实验了一下，nn_f中也涉及到了dataframe和ndarray之间的转换，但是没有出现shape的改变
        #所以..这是什么鬼？？现在的问题就是找出这样转换之后到底多了什么东西呢？？？可能只有用存入文件的办法验证了吧
        #X_split_train_df = pd.DataFrame(data=X_split_train, columns=[i for i in range(0, len(X_split_train[0]))])
        #Y_split_train_df = pd.DataFrame(data=Y_split_train, columns=[0])
        #X_split_train_df = pd.DataFrame(data=X_split_train)
        #Y_split_train_df = pd.DataFrame(data=Y_split_train)
        #X_split_train_df = pd.DataFrame(X_split_train)
        #Y_split_train_df = pd.DataFrame(Y_split_train)
        #X_noise_train, Y_noise_train = noise_augment_dataframe_data(nodes["mean"], nodes["std"], X_split_train_df, Y_split_train_df, columns=[i for i in range(0, 0)])
        
        #X_temp = X_noise_train.values
        #Y_temp = Y_noise_train.values
        #Y_list = [x for j in Y_temp for x in j]
        #Y_temp = np.ndarray(Y_list)
        
        #X_temp = X_split_train_df.values
        #Y_temp = Y_split_train_df.values
        #.reshape(1, -1)是变为一列，.reshape(-1, 1)是变为一行
        #Y_temp_reshape = Y_temp.reshape(1, -1)
        ##通过print对象的shape可以发现，X_split_train_df = pd.DataFrame(data=X_split_train)
        ##和X_split_train_df = pd.DataFrame(X_split_train)达到的效果好像真的是一样的吧？
        ##问题的关键在于为什么X_temp = X_split_train_df.values出现了shape不一致的情况？？ 
        ##但是输出到文件里面的时候明明是一样的呀，那么shape多出来的部分到底是什么东西？？
        ##哎，出现这个问题根本是因为我没搞懂shape的用法，Y_split_train.shape是(574,)
        ##但是Y_split_train_df.shape和Y_temp_reshape.shape是(574, 1)
        ##可以在控制台输出可以看到，前者是包含574个元素一个列表，后者是574个包含一个元素的列表。
        #print(X_split_train.shape)
        #print(Y_split_train.shape)
        #print(X_split_train_df.shape)
        #print(Y_split_train_df.shape) 
        #print(X_temp.shape)
        #print(Y_temp.shape)
        #print(Y_temp_reshape.shape)
        #output = pd.DataFrame(data = X_split_train)
        #output.to_csv("C:/Users/1/Desktop/X_split_train_shape.csv", index=False)
        #output = pd.DataFrame(data = X_split_train_df)            
        #output.to_csv("C:/Users/1/Desktop/X_split_train_df_shape.csv", index=False)
        #output = pd.DataFrame(data = X_temp)            
        #output.to_csv("C:/Users/1/Desktop/X_temp_shape.csv", index=False)
        #output = pd.DataFrame(data = Y_split_train)
        #output.to_csv("C:/Users/1/Desktop/Y_split_train_shape.csv", index=False)
        #output = pd.DataFrame(data = Y_split_train_df)
        #output.to_csv("C:/Users/1/Desktop/Y_split_train_df_shape.csv", index=False)
        #output = pd.DataFrame(data = Y_temp)
        #output.to_csv("C:/Users/1/Desktop/Y_temp_shape.csv", index=False)
        #output = pd.DataFrame(data = Y_temp_reshape)
        #output.to_csv("C:/Users/1/Desktop/Y_temp_reshape.csv", index=False)

        #clf.fit(X_temp.astype(np.float32), Y_temp_reshape.astype(np.longlong))
        #clf.fit(X_split_train.astype(np.float32), Y_split_train.astype(np.longlong))        
        #clf.fit(X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong))
        
        #最后花了这么多时间发现下面的写法完全不行，所以还是用更简单的办法吧
        X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_split_train, Y_split_train, columns=[i for i in range(1, 19)])
        
        clf.fit(X_noise_train.astype(np.float32), Y_noise_train.astype(np.longlong))
            
        metric = cal_nnclf_acc(clf, X_split_test, Y_split_test)
        print_nnclf_acc(metric)
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)        
    
    return best_model, best_acc

def train_nn_model_noise_validate3(nodes, X_train_scaled, Y_train, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0
    
    #这一轮就使用这一份加噪声的数据就可以了吧？没有必要在下面的for循环中也添加吧？
    #我好像真的只有用这种方式增加stacking模型之间的差异了吧？以提升泛化性能咯。
    X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_train_scaled, Y_train, columns=[i for i in range(0, 19)])

    for j in range(0, max_evals):
        
        #不对吧我在想是不是在这里面添加噪声更好一些呢，毕竟上面的噪声添加方式可能造成模型过渡拟合增加噪声之后的数据？？
        #我不知道是不是在这里面增加噪声得到的效果会更好一些呢，我觉得很郁闷问题到底出现在哪里呀？
        
        clf = NeuralNetClassifier(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(clf.module, nodes["weight_mode"], nodes["bias"])
                
        #这边的折数由5折修改为10折吧，这样子的话应该更加能够表示出稳定性吧
        skf = StratifiedKFold(Y_noise_train, n_folds=10, shuffle=True, random_state=None)
        metric = cross_val_score(clf, X_noise_train.astype(np.float32), Y_noise_train.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
    return best_model, best_acc

def train_nn_model_noise_validate4(nodes, X_train_scaled, Y_train, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0
    
    for j in range(0, max_evals):
        #不对吧我在想是不是在这里面添加噪声更好一些呢，毕竟上面的噪声添加方式可能造成模型过渡拟合增加噪声之后的数据？？
        #我不知道是不是在这里面增加噪声得到的效果会更好一些呢，我觉得很郁闷问题到底出现在哪里呀？
        X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_train_scaled, Y_train, columns=[i for i in range(0, 19)])

        
        clf = NeuralNetClassifier(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(clf.module, nodes["weight_mode"], nodes["bias"])
                
        #这边的折数由5折修改为10折吧，这样子的话应该更加能够表示出稳定性吧
        skf = StratifiedKFold(Y_noise_train, n_folds=10, shuffle=True, random_state=None)
        metric = cross_val_score(clf, X_noise_train.astype(np.float32), Y_noise_train.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
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
        
        best_model, best_acc = train_nn_model(nodes, X_split_train, Y_split_train, max_evals)
            
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

def get_oof_validate1(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_validate1(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
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

def get_oof_validate2(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_validate2(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
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

def get_oof_noise_validate1(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_noise_validate1(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
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

def get_oof_noise_validate2(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_noise_validate2(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
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

def get_oof_noise_validate3(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_noise_validate3(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
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

def get_oof_noise_validate4(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
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
        
        best_model, best_acc = train_nn_model_noise_validate4(nodes, X_split_train, Y_split_train, max_evals)
        
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

def stacked_features_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_validate1(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_validate2(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#我个人觉得这样的训练方式好像导致过拟合咯，所以采用下面的方式进行训练。
#每一轮进行get_oof_validate1的时候都增加了噪声，让每个模型都有所不同咯。
def stacked_features_noise_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
    
        #在这里增加一个添加噪声的功能咯
        X_noise_train, Y_noise_train = noise_augment_dataframe_data(nodes_list[0]["mean"], nodes_list[0]["std"], X_train_scaled, Y_train, columns=[i for i in range(1, 20)])#columns=[])

        oof_train, oof_test, best_model= get_oof_noise_validate1(nodes_list[i], X_noise_train.values, Y_noise_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#下面是我想到的第二种增加模型噪声的方式以防止过拟合咯。
#我个人觉得这样的训练方式好像导致过拟合咯，所以采用下面的方式进行训练。
#每一轮进行get_oof_validate1的时候都增加了噪声，让每个模型都有所不同咯。
def stacked_features_noise_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate2(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_noise_validate3(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate3(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_noise_validate4(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate4(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#这个就是一个单节点神经网络预测咯，用下面的方法试试水咯
def nn_predict(best_nodes, X_train_scaled, Y_train, X_test_scaled, folds=10, max_evals=50):

    best_acc = 0.0
    best_model = 0.0
    
    for j in range(0, max_evals):
        #不对吧我在想是不是在这里面添加噪声更好一些呢，毕竟上面的噪声添加方式可能造成模型过渡拟合增加噪声之后的数据？？
        #我不知道是不是在这里面增加噪声得到的效果会更好一些呢，我觉得很郁闷问题到底出现在哪里呀？

        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module = create_nn_module(best_nodes["input_nodes"], best_nodes["hidden_layers"], 
                                                         best_nodes["hidden_nodes"], best_nodes["output_nodes"], best_nodes["percentage"]),
                                  max_epochs = best_nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=best_nodes["patience"])],
                                  device = best_nodes["device"],
                                  optimizer = best_nodes["optimizer"]
                                  )
        init_module(clf.module, best_nodes["weight_mode"], best_nodes["bias"])
                
        #这边的折数由5折修改为10折吧，这样子的话应该更加能够表示出稳定性吧
        skf = StratifiedKFold(Y_train, n_folds=folds, shuffle=True, random_state=None)
        metric = cross_val_score(clf, X_train_scaled.astype(np.float32), Y_train.astype(np.longlong), cv=skf, scoring="accuracy").mean()
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.longlong))
    
    acc = cal_nnclf_acc(best_model,  X_train_scaled, Y_train)
    print_nnclf_acc(acc)
    
    save_best_model(best_model, best_nodes["title"])
    Y_pred = best_model.predict(X_test_scaled.astype(np.float32))        
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
    output.to_csv(best_nodes["path"], index=False)
    print("prediction file has been written.")
            
    return best_model, best_acc

#这个选择最佳模型的时候存在过拟合的风险
def nn_stacking_predict(best_nodes, data_test, stacked_train, Y_train, stacked_test, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0

    #我已经将这份代码的best_nodes["title"]由原来的titanic改为stacked_titanic作为新版本
    if (exist_files(best_nodes["title"])):
        #在这里暂时不保存stakced_train以及stacked_test吧
        best_model = load_best_model(best_nodes["title"]+"_"+str(len(nodes_list)))
        best_acc = cal_nnclf_acc(best_model, stacked_train.values, Y_train.values)
         
    for i in range(0, max_evals):
        
        #这边不是很想用train_nn_model代替下面的函数代码
        #因为这下面的代码还涉及到预测输出的问题不好修改
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module = create_nn_module(stacked_train.columns.size, best_nodes["hidden_layers"], 
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
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred
   
#lr没有超参搜索而且没有进行过cv怎么可能会取得好成绩呢？ 
def lr_stacking_predict(best_nodes, data_test, stacked_train, Y_train, stacked_test, max_evals=50):
    
    best_acc = 0.0
    best_model = 0.0
       
    #这里并不需要保存最佳的模型吧，只需要将stacked_train之类的数据记录下来就行了
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        #这边是不是需要加入一些随机化的因素或者其他因素？？
        clf = LogisticRegression()        
        clf.fit(stacked_train, Y_train)
        
        metric = cal_nnclf_acc(clf, stacked_train.values, Y_train.values)
        print_nnclf_acc(metric)
        
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
    
        if (flag):
            #这个版本的best_model终于是全局的版本咯，真是开森呢。。
            save_best_model(best_model, best_nodes["title"]+"_"+str(len(nodes_list)))
            Y_pred = best_model.predict(stacked_test.values.astype(np.float32))
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            
            output.to_csv(nodes_list[0]["path"], index=False)
            print("prediction file has been written.")
        print()
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred

#lr进行了超参搜索选出最好的结果进行预测咯 
def lr_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, max_evals=2000):
    
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

    save_best_model(random_search.best_estimator_, nodes_list[0]["title"]+"_"+str(len(nodes_list)))
    Y_pred = random_search.best_estimator_.predict(stacked_test.values.astype(np.float32))
            
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(nodes_list[0]["path"], index=False)
    print("prediction file has been written.")
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return random_search.best_estimator_, Y_pred

def tpot_stacking_predict(best_nodes, data_test, stacked_train, Y_train, stacked_test, generations=100, population_size=100):
    
    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity = 2)
    tpot.fit(stacked_train, Y_train)
    best_acc = tpot.score(stacked_train, Y_train)
    Y_pred = tpot.predict(stacked_test)
    best_model = tpot
         
    save_best_model(best_model.fitted_pipeline_, best_nodes["title"]+"_"+"tpot")
    Y_pred = best_model.predict(stacked_test)
            
    data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(best_acc["path"], index=False)
    print("prediction file has been written.")
            
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return best_model, Y_pred
    
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
         "input_nodes":hp.choice("input_nodes", [20]),
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
               "input_nodes":[20],
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
              "input_nodes":20,
              "hidden_layers":3, 
              "hidden_nodes":60, 
              "output_nodes":2,
              "percentage":0.15,
              "weight_mode":1,
              "bias":0.0,
              "device":"cpu",
              "optimizer":torch.optim.Adam
              }

#那么自助法应该就是我最后一次在修改代码了吧
#我之前觉得可能stacking的每个分类器用自助法
#但是这样应该不太可能行吧，感觉明显不太靠谱
#所以我最后将噪声和交叉验证进行结合咯，希望有用吧。。
#我理解问题在于单模型的时候就过拟合了，所以必须加噪声。。
#这个版本追平了我之前的最好记录，但是呢也才仅仅0.79904正确率而已。
#我现在好像真的找不到修改的方向了吧，试一下五个不同节点呢，我觉得应该差不多的吧
"""
start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
#nodes_list = parse_trials(trials, space_nodes, 3)
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
for item in nodes_list:
    item["device"] = "cuda"
    item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_train_scaled, Y_train, X_test_scaled, 50, 20)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")
lr_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 2000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#mmp 是折数太低了还是啥请款呀，居然在leaderboard上面只有0.74162的分数= =！
#好像真的每次不同节点的结果都是比不上相同节点的结果呢，这是为什么呢
start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
nodes_list = parse_trials(trials, space_nodes, 5)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
for item in nodes_list:
    item["device"] = "cuda"
    item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_train_scaled, Y_train, X_test_scaled, 10, 20)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")
lr_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 2000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#现在我准备做最后一个实验咯，就是集中不同的方式进行比较咯
train_acc = []
valida_acc = []
time_cost = []

start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
 
for i in range(0, 3):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.15, stratify=Y_train)

    #1个节点的不添加噪声的版本
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate1(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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

    start_time = datetime.datetime.now()
    nodes_list = [best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate2(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate4(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    #3个节点的不添加噪声的版本
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate1(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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

    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate2(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate4(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    #5个节点的不添加噪声的版本
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate1(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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

    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_validate2(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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
    
    start_time = datetime.datetime.now()
    nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
    for item in nodes_list:
        item["device"] = "cuda"
        item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
    stacked_train, stacked_test = stacked_features_noise_validate4(nodes_list, X_split_train, Y_split_train, X_split_test, 5, 20)
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

for i in range(0, len(train_acc)):
    print(train_acc[i])
    print(valida_acc[i])

for i in range(0, len(time_cost)):
    print(time_cost[i])
"""

"""
#我在想是否我不需要stacking呀，我理解同节点模型stacking之间差异不大呀
#而且不同节点之间stacking效果反而下降了。。我是真的去了咯。。。应该是单模型和
#是不是我不使用stacking直接使用单神经网络模型+交叉验证就能够得到不错的分数呢？
#我现在觉得问题应该出现在单模型身上，单模型不够强造成的现在这样子的结果咯
#所以以后我觉得应该不需要在nn_f中添加噪声了吧。我的计算条件有限stacking暂时改用两三个节点就好。
#或者以后要么直接玩单模型，我现在使用了2个节点每次计算50次最后得到的结果在leaderboard上只有0.78947的正确率扎心了
start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
nodes_list = parse_trials(trials, space_nodes, 2)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
for item in nodes_list:
    item["device"] = "cpu"
    item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
stacked_train, stacked_test = stacked_features_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, 10, 50)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")
lr_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 2000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#现在准备构建一个单模型的分类器吧，我觉得是不是只要随机初始足够的多应该可以获得更好的结果吧？
#我感觉可能就是这么回事儿吧，但是我现在没办法访问github和kaggle家里面台式机也出现问题了，我真的觉得心很累
#mmp,公司的网络现在开始屏蔽了很多网站我都上不去github和kaggle咯，今天老子花了很多时间最后发现唯一的房还是就是用热点上去
#15节点700次的计算在leaderboarder上面只有0.79425的正确率，我感觉开始有点怀疑人生了
#只有尝试最后一个版本咯，那就是单节点然后进行3000次的计算得到最后的结果，或者尝试删除离群点。
#其实我觉得leaderboard上面0.82和0.799没啥太大区别，就是前者比后者多正确了五个而已，所以删除离群点可能很有用吧。
#我对这个3000次测试的版本也不是特别的有信心，反倒是觉得可能删除离群点作用应该更大一些的吧。
#哇塞这个3000次的计算版本果然还是不太行呀只有0.78947的正确率，所以最后试试离群点删除吧，真的不打算在做咯。
start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
best_nodes["device"] = "cpu"
best_nodes["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
#nodes_list = parse_trials(trials, space_nodes, 2)
#nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
#for item in nodes_list:
#    item["device"] = "cpu"
#    item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
nn_predict(best_nodes, X_train_scaled.values, Y_train.values, X_test_scaled.values, 15, 3000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

#哇塞，使用噪声在leaderborad上面才只有0.76的准确率，可能还是噪声误导了学习器
#看来增加噪声并不等效于排除离群点，之后要做的事情就是按照之前的文档重新写一遍程序咯。
#我最后的实验应该是准备尝试新的思路整理一下数据咯。
start_time = datetime.datetime.now()
files = open("titanic_intermediate_parameters_2018-12-18194719.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
#nodes_list = parse_trials(trials, space_nodes, 3)
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
for item in nodes_list:
    item["device"] = "cpu"
    item["path"] = "C:/Users/1/Desktop/Titanic_Prediction.csv"
stacked_train, stacked_test = stacked_features_noise_validate3(nodes_list, X_train_scaled, Y_train, X_test_scaled, 15, 32)
save_stacked_dataset(stacked_train, stacked_test, "stacked_titanic")
lr_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 2000)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))