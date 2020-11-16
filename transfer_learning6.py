"""
Search for a good model for the [Titanic](https://www.kaggle.com/c/titanic) dataset.
First, you need to download the titanic dataset file
[train.csv](
https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/titanic/train.csv
)
and
[eval.csv](
https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/titanic/eval.csv
).
Second, replace `PATH_TO/train.csv` and `PATH_TO/eval.csv` in the following example
with the real path to those two files.
Then, you can run the code.
"""

"""
这是原版的代码按照注释修改了可以运行，但是找到的网络输出准确率都只有0.61的样子基本是没用的，可能是输入输出写法有问题？
import autokeras as ak

# Initialize the classifier.
clf = ak.StructuredDataClassifier(max_trials=100)
# x is the path to the csv file. y is the column name of the column to predict.
clf.fit(x='kaggle_titanic_files/train.csv', y='Survived')
# Evaluate the accuracy of the found model.
accuracy = clf.evaluate(x='kaggle_titanic_files/train.csv', y='Survived')
print('Accuracy: {0}'.format(accuracy))
"""

#之前程序无法运行果然是代码的问题，我自己修改了一下代码就可以正常输出了
#现在的问题是貌似输出的不是最优解呢，
import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


# load train data and test data
data_train = pd.read_csv("kaggle_titanic_files/train.csv")
data_test = pd.read_csv("kaggle_titanic_files/test.csv")
# combine train data and test data
combine = [data_train, data_test]

# create title feature from name feature
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Capt'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# create familysize from sibsp and parch feature
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

# 现在这里的性别不能够被替换否则下面无法使用DictVectorizer进行映射
# switch sex to number, which could make following missing age prediction easier
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# predict missing age
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]
    dataset['Age'] = dataset['Age'].astype(int)

# 为了之后使用dictvector进行映射，现在又将性别变为字符串
# 如果不用这样的操作变为字符串就无法用dictvector替换啦
# 因为dictvector的替换就是直接将非数值的东西变成单独的一类
# switch sex to string for later DictVectorizer
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({1: 'female', 0: 'male'})

# 这里的mode是求解pandas.core.series.Series众数的第一个值（可能有多个众数）
# fill missing data in embarked feature
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 将data_test中的fare元素所缺失的部分由已经包含的数据的中位数决定哈
# fill missing data in fare
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

# switch cabin feature into "cabin" or "no cabin"
for dataset in combine:
    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = "cabin"
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = "no cabin"

# switch number of pclass into string for later DictVectorizer
for dataset in combine:
    dataset.loc[dataset['Pclass'] == 1, 'Pclass'] = "1st"
    dataset.loc[dataset['Pclass'] == 2, 'Pclass'] = "2nd"
    dataset.loc[dataset['Pclass'] == 3, 'Pclass'] = "3rd"

# 尼玛给你说的这个是共享船票，原来的英文里面根本就没有这种说法嘛
# find if there is the same ticket, which means higher probability of survival
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
# print(df)
df_ticket = df.index.values  # 共享船票的票号
tickets = data_train.Ticket.values  # 所有的船票
# print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0  # 遍历所有船票，在共享船票里面的为1，否则为0
    result.append(ticket)
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values  # 共享船票的票号
tickets = data_train.Ticket.values  # 所有的船票

# create ticket_count feature in training data
result = []
for ticket in tickets:
    if ticket in df_ticket:
        # 主要是为了使用DictVectorizer类映射所以改写下面的样子
        # ticket = 1
        ticket = "share"
    else:
        # ticket = 0                   #遍历所有船票，在共享船票里面的为1，否则为0
        ticket = "no share"
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_train = pd.concat([data_train, results], axis=1)

# create ticket_count feature in test data
df = data_test['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values
tickets = data_test.Ticket.values
result = []
for ticket in tickets:
    if ticket in df_ticket:
        # ticket = 1
        ticket = "share"
    else:
        # ticket = 0
        ticket = "no share"
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_test = pd.concat([data_test, results], axis=1)

# data_train_1 data_train data_test_1 data_test have different id
data_train_1 = data_train.copy()
data_test_1 = data_test.copy()
data_test_1 = data_test_1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1)

# get feature from data_train_1 and data_test_1
X_train = data_train_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]
Y_train = data_train_1['Survived']
X_test = data_test_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]

# combine X_train and X_test
X_all = pd.concat([X_train, X_test], axis=0)
# print(X_all.columns)
# 下面是我补充的将性别、姓名、Embarked修改为了one-hot编码类型了
# 原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
# use DictVectorizer to create something like one-hot encoding
dict_vector = DictVectorizer(sparse=False)
X_all = dict_vector.fit_transform(X_all.to_dict(orient='record'))
X_all = pd.DataFrame(data=X_all, columns=dict_vector.feature_names_)

# scale feature to (0,1), which could make process of training easier
X_all_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns=X_all.columns)
X_train_scaled = X_all_scaled[:len(X_train)]
Y_train = Y_train[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]



# Initialize the classifier.
clf = ak.StructuredDataClassifier(max_trials=3)
# x is the path to the csv file. y is the column name of the column to predict.
clf.fit(x=X_train_scaled, y=Y_train)
# Evaluate the accuracy of the found model.
clf = clf.load_searcher().load_best_model().produce_keras_model()
accuracy = clf.evaluate(x=X_train_scaled, y=Y_train)
print('Accuracy: {0}'.format(accuracy))