#coding=utf-8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
"""
import os
print(os.listdir("../input"))
"""

# Any results you write to the current directory are saved as output.

import numpy as np 
import pandas as pd

dataset = pd.read_csv("C:/Users/win7/Desktop/train.csv")
X_test = pd.read_csv("C:/Users/win7/Desktop/test.csv")
#i.split(',')[1]的含义是将i用','划分成两个部分并选择后一部分。strip()清除前面的空格
#我真的觉得Python的代码看起来花里胡哨，但是仔细分析体察感觉确实也算是灵活简洁咯。
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title'] = pd.Series(dataset_title)
dataset['Title'].value_counts()
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')

dataset_title = [i.split(',')[1].split('.')[0].strip() for i in X_test['Name']]
X_test['Title'] = pd.Series(dataset_title)
X_test['Title'].value_counts()
X_test['Title'] = X_test['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')

dataset['FamilyS'] = dataset['SibSp'] + dataset['Parch'] + 1
X_test['FamilyS'] = X_test['SibSp'] + X_test['Parch'] + 1

def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'

#apply函数可以对DataFrame对象进行操作，既可以作用于一行或者一列的元素，也可以作用于单个元素。
dataset['FamilyS'] = dataset['FamilyS'].apply(family)
X_test['FamilyS'] = X_test['FamilyS'].apply(family)

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
X_test['Embarked'].fillna(X_test['Embarked'].mode()[0], inplace=True)
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
X_test['Age'].fillna(X_test['Age'].median(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True)

dataset = dataset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
X_test_passengers = X_test['PassengerId']
X_test = X_test.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
#iloc函数两个参数的含义分别是多少行，多少列的数据
X_train = dataset.iloc[:, 1:9].values
Y_train = dataset.iloc[:, 0].values
X_test = X_test.values

# Converting the remaining labels to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#我已经看到很多的代码使用LabelEncoder, OneHotEncoder
#或许神经网络一般就是要使用这个的吧，晚点在我自己的代码上试一哈这个
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])
X_train[:, 6] = labelencoder_X_1.fit_transform(X_train[:, 6])
#labelencoder执行之前的结果是
#array([[3, 'male', 22.0, ..., 'S', 'Mr', 'Couple'],
#[1, 'female', 38.0, ..., 'C', 'Mrs', 'Couple'],
#[3, 'female', 26.0, ..., 'S', 'Miss', 'Single'],
#..., 
#[3, 'female', 28.0, ..., 'S', 'Miss', 'InterM'],
#[1, 'male', 26.0, ..., 'C', 'Mr', 'Single'],
#[3, 'male', 32.0, ..., 'Q', 'Mr', 'Single']], dtype=object)
#labelencoder执行之后的结果是
#array([[3, 1, 22.0, ..., 2, 2, 0],
#[1, 0, 38.0, ..., 0, 3, 0],
#[3, 0, 26.0, ..., 2, 1, 3],
#..., 
#[3, 0, 28.0, ..., 2, 1, 1],
#[1, 1, 26.0, ..., 0, 2, 3],
#[3, 1, 32.0, ..., 1, 2, 3]], dtype=object)
#这个方式还是有点意思的，可以积累一下子呢。

labelencoder_X_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])
X_test[:, 6] = labelencoder_X_2.fit_transform(X_test[:, 6])

# Converting categorical values to one-hot representation
one_hot_encoder = OneHotEncoder(categorical_features = [0, 1, 4, 5, 6])
X_train = one_hot_encoder.fit_transform(X_train).toarray()
X_test = one_hot_encoder.fit_transform(X_test).toarray()

from sklearn.model_selection import train_test_split

#看到这里我感觉有点绝望了呀，因为至此为止我并没有发现任何的异常呢。
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 270)
        self.fc2 = nn.Linear(270, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        
        return x
    
net = Net()

batch_size = 50
num_epochs = 50
learning_rate = 0.001
batch_no = len(x_train) // batch_size

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

from sklearn.utils import shuffle
from torch.autograd import Variable

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch+1))
    x_train, y_train = shuffle(x_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss =criterion(ypred_var, y_var)
        loss.backward()
        optimizer.step()
        
# Evaluate the model
test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
with torch.no_grad():
    result = net(test_var)
values, labels = torch.max(result, 1)
num_right = np.sum(labels.data.numpy() == y_val)
print('Accuracy {:.2f}'.format(num_right / len(y_val)))

# Applying model on the test data
X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=True) 
with torch.no_grad():
    test_result = net(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()

def ff():
    print("Hello World")
    
"""
import csv

submission = [['PassengerId', 'Survived']]
for i in range(len(survived)):
    submission.append([X_test_passengers[i], survived[i]])
    
with open('submission.csv', 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)
    
print('Writing Complete!')
"""
