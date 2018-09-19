#coding=utf-8
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

import numpy as np
import pandas as pd
import autokeras as ak
from sklearn import preprocessing

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
# Load data
df_train, df_test = get_boston_dataset()

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']
column_descriptions = {
  'MEDV': 'output'
  , 'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

# Score the model on test data
test_score = ml_predictor.score(df_test, df_test.MEDV)

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()

trained_model = load_ml_model(file_name)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predictions = trained_model.predict(df_test)
print(predictions)
"""

#这个库唯一的安慰就是效果比不上我的神经网络么。。
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, brier_score_loss, accuracy_score


x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.2)
x_y_train_scaled = pd.concat([x_train, y_train], axis=1)
x_y_test_scaled = pd.concat([x_test, y_test], axis=1)
X_Y_train_scaled = pd.concat([X_train_scaled, Y_train], axis=1)
#X_Y_train_scaled.to_csv("C:/Users/1/Desktop/X_Y_train_scaled.csv", encoding="utf-8", index=False)

column_descriptions = {
    "Survived": "output",
    #这个categorical到底有什么用处啊，好像直接注释掉也没关系的嘛
    #"Pclass": "categorical",
    #"Sex": "categorical",
    #"Age": "categorical",
    #"Fare": "categorical",
    #"Embarked": "categorical",
    #"Cabin": "categorical",
    #"Title": "categorical",
    #"FamilySizePlus": "categorical",
    #"Ticket_Count": "categorical",
}

"""
#这些B模型里面的输出了一大堆信息，看得老子头晕眼花的，不知道默认把verbose关闭了吗
ml_predictor = Predictor(type_of_estimator="classifier", column_descriptions=column_descriptions, verbose=False)

#经过我的查阅，我发现train函数中的_scorer=None, scoring=None
#下面分别设置了_scorer和scoring的方式进行实验，第一种方式似乎不行呀
#ml_predictor.train(x_y_train_scaled, scoring="accuracy")
#还有这里的cv默认设置居然是2这个有点奇葩吧，我觉得正常都应该是5才行吧
#ml_predictor.train(x_y_train_scaled, _scorer=accuracy_score, cv=5, verbose=False)
#这个模型其实执行之后将输出一些负数和准确率，这的库给出的准确率完全没有TPOT惊艳
#虽然我并没有设置，然而他还是把什么混淆矩阵或者是Buckets Edges之类的东西输出了，虽然我看不懂
#我将cv的值由2修改为5、10、20准确率似乎并没有什么提升的样子，看来这个可能是唯一让我欣慰的东西吧
ml_predictor.train(X_Y_train_scaled, _scorer=accuracy_score, cv=20, verbose=False)


#这个现在倒是可以运行了，但是尼玛的输出的是什么东西哦，所以现在需要输出准确率的结果吧
#然后我现在需要看懂这个框架每一步的输出到底是什么，我现在看到这些输出非常的困惑
#等这个准确率的结果搞定了之后就准备测试深度学习的框架吧，其实这个例子修改并不难的吧
#主要是心态问题吧，一直在学习这些东西，感觉被这些库都给搞烦了。。其实还是比较简单，现在主要深度学习
print(ml_predictor.score(X_Y_train_scaled, X_Y_train_scaled.Survived, verbose=0))
"""

#我现在来试一哈深度学习的模型效果到底如何呢？让我们拭目以待，此时此刻我的心情还有点忐忑呢
#测试结果发现，这个库的效果其实挺差的，感觉被skorch完爆了，skorch还可以设计不同的细节个性化的个性化模型
#也许这些东西的效果可以从star数目上得以体现吧。然后阅读了剩下的文档也没有发现一些奇怪的东西咯
#skorch可以完成各种模型设计但是auto_ml不行，skorch还可以配合超参搜索做数据集增强这个不行
#skorch的文档比较好简洁易用hyperopt可以简化今后操作这个库使用还是比较简单的但是接口做的很敷衍。。
#我觉得现在的出路可能就在遗传算法之类的计算神经网络的结构吧这个算法估计得我自己来设计咯
#不过话也说回来，如果设计得好那么我的机器学习生涯也有了第一个亮点咯。还有可能设计神经网络从成熟和层节点来搜索咯
ml_predictor = Predictor(type_of_estimator="classifier", column_descriptions=column_descriptions)

ml_predictor.train(X_Y_train_scaled, _scorer=accuracy_score, cv=20, verbose=False, model_names=['DeepLearningClassifier'])

#这个深度学习的模型不支持计算准确率或者说不支持sklearn的准确率计算方式，看来这个问题倒是和skorch一样咯
#ml_predictor.score(X_Y_train_scaled, cv=20, verbose=False)

