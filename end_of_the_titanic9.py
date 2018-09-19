#coding=utf-8
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


import os
import csv
import pickle

TRAIN_IMG_DIR = "C:/Users/1/Desktop/re/train"
TRAIN_CSV_DIR = "C:/Users/1/Desktop/re/train_labels.csv"
TEST_IMG_DIR = "C:/Users/1/Desktop/re/test"
TEST_CSV_DIR = "C:/Users/1/Desktop/re/test_labels.csv"

def mkcsv(img_dir, csv_dir):
    list = []
    list.append(['File Name','Label'])
    for file_name in os.listdir(img_dir):
        if file_name[0] == '3':   #bus
            item = [file_name, 0]
        elif file_name[0] == '4': #dinosaur
            item = [file_name, 1]
        elif file_name[0] == '5': #elephant
            item = [file_name, 2]
        elif file_name[0] == '6': #flower
            item = [file_name, 3]
        else:
            item = [file_name, 4] #horse
        list.append(item)

    print(list)
    f = open(csv_dir, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(list)

mkcsv(TRAIN_IMG_DIR, TRAIN_CSV_DIR)
mkcsv(TEST_IMG_DIR, TEST_CSV_DIR)


from tensorflow.keras.preprocessing import image
import os


def format_img(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        path_name = os.path.join(input_dir, file_name)
        img = image.load_img(path_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        path_name = os.path.join(output_dir, file_name)
        img.save(path_name)


from autokeras.classifier import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array

import numpy as np
import pandas as pd

TRAIN_CSV_DIR = "C:/Users/1/Desktop/re/train_labels.csv"
TRAIN_IMG_DIR = "C:/Users/1/Desktop/re/train"
TRAIN_IMG_DIR_SCALED = "C:/Users/1/Desktop/re/train_scaled"
TEST_CSV_DIR = "C:/Users/1/Desktop/re/test_labels.csv"
TEST_IMG_DIR = "C:/Users/1/Desktop/re/test"
TEST_IMG_DIR_SCALED = "C:/Users/1/Desktop/re/test_scaled"

PREDICT_IMG_PATH = "C:/Users/1/Desktop/re/test_scaled/719.jpg"

MODEL_DIR = "C:/Users/1/Desktop/re/model/my_model.h5"
MODEL_PNG = "C:/Users/1/Desktop/re/model/model.png"
IMAGE_SIZE = 28

if __name__ == '__main__':

    #train_data = train_data.astype('float32') / 255
    #test_data = test_data.astype('float32') / 255
    #这个好像真的是没办法解决一般的，图像读出的像素结构蛮奇怪的
    train_data = X_train_scaled
    train_labels = Y_train

    print("train data shape:", train_data.shape)

    # 使用图片识别器
    clf = ImageClassifier(verbose=True)

    clf.fit(train_data, train_labels, time_limit=1 * 60)
    
    """
    # 找到最优模型后，再最后进行一次训练和验证
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)
    # 给出评估结果
    #顺便看了一下classifier里面的evaluate、fit和final_fit函数
    #我个人感觉这些函数的接口设计的都是蛮奇怪的吧，也许可能是因为我不太懂深度学习？？
    y = clf.evaluate(test_data, test_labels)
    print("evaluate:", y)
    y = clf.predict(x)
    print("predict:", y)

    #算了还是直接使用pickle进行封装吧
    files = open("autokeras_best_model.pickle", "wb")
    pickle.dump(clf, files)
    files.close()
    
    files = open("autokeras_best_model.pickle", "rb")
    best_model = pickle.load(files)
    files.close()
    """