#coding=utf-8
#这次才发现hpsklearn其实是可以支持超参搜索空间的，那我之前end_of_the_titanic5
#的工作可能需要进行一些返工了吧，他妈的一个多星期之后由于处理特征才回顾到这个问题。
#不对，我之前的工作并没有白做，因为skorch的模型没有score等，只有我自己实现
#这个hpsklearn作用主要在于（1）搜索模型。（2）搜索超参。（1）对我没用（2）hyperopt一样能做
#所以这次能够确定hpsklearn对我确实是没有什么帮助哈。
#所以以后确定这个库能为我做什么的时候必须反复研读文档、教程和源代码才能确定的吧。。
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np
from mistune import preprocessing
from sklearn import preprocessing

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skorch import NeuralNetClassifier
import sys
import skorch
import pandas as pd
import torch.nn.init

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

#我刚一运行就看到hpsklearn报出下面的错，我当时还一头雾水为毛是openblas，我都不知道openblas是啥呢
#if you are using openblas set OMP_NUM_THREADS=1 or risk subprocess calls hanging indefinitely
#我谷歌了一下上面的报错大概是下面的意思，个人感觉暂时不太需要处理这个问题咯
#The first warning is letting you know that if you are using a particular linear algebra library
#(OpenBLAS) you could run into problems. I've never seen it be an issue, but you can run export 
#OMP_NUM_THREADS=1 just in case and it will make the warning go away.
#The second warning is saying a few modules in sklearn are getting moved around in the future. 
#It doesn't affect anything right now, but I'll have to update it before version 0.20 of sklearn.
#Download the data and split into training and test sets
if __name__ == "__main__":

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
    #将X_train与X_test写入到文件中查看是否存在异样，现在两个对象内的数据明显是一样的呀
    #pd.DataFrame(data=X_train).to_csv("C:\\Users\\win7\\Desktop\\X_train.csv", index=False)
    #pd.DataFrame(data=X_test).to_csv("C:\\Users\\win7\\Desktop\\X_test.csv", index=False)
    #原来这样的写法是横向拼接，X_test只有400多个和891个数据的X_train拼接，后面400多数据必然出现NAN
    #X_all = pd.concat([X_train, X_test], axis=1)
    #pd.DataFrame(data=X_all).to_csv("C:\\Users\\win7\\Desktop\\X_all.csv", index=False)
    X_all = pd.concat([X_train, X_test], axis=0)

    #我觉得训练集和测试集需要在一起进行特征缩放，所以注释掉了原来的X_train的特征缩放咯
    X_all_scaled = pd.DataFrame(preprocessing.scale(X_all), columns = X_train.columns)
    #X_train_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)
    X_train_scaled = X_all_scaled[:len(X_train)]
    X_test_scaled = X_all_scaled[len(X_train):]

    """
    iris = load_iris()

    X = iris.data
    y = iris.target

    test_size = int(0.2 * len(y))

    #刚才点了一下np.random.permutation发现里面全是乱码原来是pyd文件咯
    np.random.seed(13)
    #np.random.permutation的意思就是对len(X)以内的数据进行随机排列
    #原来还可以这个样子写随机排序感觉还蛮优雅的样子呢
    indices = np.random.permutation(len(X))
    #X_train = X[ indices[:-test_size]]可以这样赋值的吗，看来直接给序号就行？
    X_train = X[ indices[:-test_size]]
    y_train = y[ indices[:-test_size]]
    X_test = X[ indices[-test_size:]]
    y_test = y[ indices[-test_size:]]
    """

    #hyperopt中将搜索控件定义为space,搜索空间中每一个值即为图上的一个节点
    #hyperopt.pyll即为处理这种图上节点的类，这种图也被称为hyperopt.pyll图
    #这个文档对于hyperopt介绍的蛮清楚的https://github.com/hyperopt/hyperopt/wiki/FMin
    #比如说hp.randint(label, upper)其实是返回0到upper之间的随机整数（源代码装饰器感觉没看懂）
    #hp.quniform(label, low, high, q)返回round(uniform(low, high)/q) * q的结果
    #以及hp.qloguniform(label, low, high, q)返回round(exp(uniform(low, high))/q) * q
    #这些都是我需要的，之前有花一些时间去查询这些但是之前的文档都没这些内容代码和注释也很渣

    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    #这个第一个参数如果为空的时候将会随机加入一个预处理这也太秀了吧
    #第一个参数是preprocessing也就是sklearn中的preprocessing.scale、pca、one_hot_encoder之类的
    #any_class是所有分类器的集合好像有点炫酷的样子，但是应该不能够支持skorch模型吧，那我用这个有个鸡巴用呀。。
    #classifier可以提供多个分类器一起进行，还可以以概率的形式选择分类器，感觉分类器还是得神经网络，毕竟上限更高。。
    #hpsklearn上面文档说明有：当你不知道使用什么学习器的时候可以尝试使用hpsklearn找到最优解，我觉得这个是个伪需求。。
    #algo即为该模型所支持的超参搜索算法，有下面这些算法Random Search、TPE、Annealing、Tree、Gaussian Process Tree
    #我不知道ex_preprocs、fit_increment、之类的东西到底有什么卵用处，同时放入regressor可能是为了
    estim = HyperoptEstimator(#classifier=any_classifier('my_clf'),#这样尼玛的写法可能用了线性分类器，
                              classifier=net,
                              #尼玛的崽种，官方文档贴出来的例子不能直接运行，居然是没有导入头文件，害得我google调试了一下才反应过来。。
                              #这个any_preprocessing居然是将所有的preprocessing一起使用的吗，这尼玛币也太煞笔了吧，一点也不优雅。。
                              #preprocessing=any_preprocessing('my_pre'),这个库设计的有点不科学，根据文档说明我不提供preprocessing居然会随便预处理，我只有提供空了。。
                              #preprocessing=any_preprocessing('my_pre'),
                              preprocessing=[],
                              algo=tpe.suggest,
                              max_evals=100,
                              trial_timeout=120)

    # Search the hyperparameter space based on the data
    #运行下面的代码又出现了下面的错误提示，果然又是多线程的问题
    #The "freeze_support()" line can be omitted if the program
    #is not going to be frozen to produce an executable.
    #果然是因为这个estimator采用了进程池的缘故，我本来想设置n_jobs
    #但是API并没有提供这个选项而且我查到多线程的部分了但是不知道怎么修改
    #X_test_ = pd.DataFrame(data=X_test)#这里能打断点呀，不知道为毛之前不能打断点
    #y_test_ = pd.DataFrame(data=y_test)#之前有过一次不能打断点的情况，我不记得是怎么处理的咯
    estim.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong))

    Y_pred = estim.predict(X_train_scaled.values.astype(np.float32))
    count = (Y_pred == Y_train).sum()
    print("预测正确率",count/len(Y_pred))
    #因为net并没有score函数所以只能够用上面的写法咯
    #estim.score(X_train_scaled, Y_train)
    
    """
    # Show the results
    #print(estim.score(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)))
    Y_pred = net.predict(X_train_scaled.values.astype(np.float32))
    count = (Y_pred == Y_train).sum()
    print(count/len(Y_pred))
    """

    #现在终于可以运行了，但是出现了非常尼玛币煞笔的东西：满屏幕的提示，唯独没有中间的运行数据显示。
    #第二个提示也就算了吧，一直都有的。但是第一个提示是因为使用了线性模型的缘故吗，想办法去掉并显示中间结果吧。
    #原来可以不用考虑去掉提示线性模型的，毕竟我以后主要是用于选择神经网络的部分咯，不是线性模型就还好
    #If you are using openblas if you are using openblas set OMP_NUM_THREADS=1 or risk subprocess calls hanging indefinitely
    #D:\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version
    #0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that
    #the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
    #"This module will be removed in 0.20.", DeprecationWarning)

    #果然不出我所料，skorch实现了sklearn的相关接口就能够用于hpsklearn的超参选择咯，不过选择自己实现类似score的功能
    #所以用这个库还是有点卵用的，至少可以计算出准确率以及可以选出最佳模型，但是这个居然没有space，这尼玛的是在逗我吧
    #如果并没有超参搜索那么我用这个有什么用呢。。所以最后还是回到了原来的问题咯，修改原框架并克隆其学习器吧。。。
    #但是hpsklearn仅仅能够做到这个程度确实是有点弱了吧。。。
    print(estim.best_model())