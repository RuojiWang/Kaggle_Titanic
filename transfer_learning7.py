# https://www.jianshu.com/p/b26cbd587dd6 这是一个最简单nni的titanic的例子
# 原来autokeras这么的不堪https://www.msra.cn/zh-cn/news/features/nni，垃圾东西浪费老子时间
import nni
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import pickle

"""
x_train = pickle.load(open("./data/x_train", 'rb'))
x_test = pickle.load(open("./data/x_test", 'rb'))
test = pd.read_csv("./data/test.csv")
train = pd.read_csv("./data/train.csv")


train = pd.read_csv("kaggle_titanic_files/train.csv")
test = pd.read_csv("kaggle_titanic_files/test.csv")
x_train = pd.read_csv("kaggle_titanic_files/train.csv")
x_test = pd.read_csv("kaggle_titanic_files/test.csv")
"""

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
X_train_scaled = X_all_scaled[:len(X_train)] #因为训练数据总数为891将其改变为880，以避免十折交叉验证的时候batchnorm的batchsize为1
Y_train = Y_train[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]


passengerId = test['PassengerId']
y_train = train['Survived'].ravel()

# 获取默认参数
def get_default_parameters():
     params = {
          'learning_rate': 0.02,
          'n_estimators': 2000,
          'max_depth': 4,
          'min_child_weight':2,
          'gamma':0.9,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'objective':'binary:logistic',
          'nthread':-1,
          'scale_pos_weight':1
     }
     return params

# 获取模型
def get_model(PARAMS):
     model = xgb.XGBClassifier()
     model.learning_rate = PARAMS.get("learning_rate")
     model.max_depth = PARAMS.get("max_depth")
     model.subsample = PARAMS.get("subsample")
     model.colsample_btree = PARAMS.get("colsample_btree")
     return model

# 运行模型
kf = KFold(n_splits=5)
def run(x_train, y_train, model):
     scores = cross_val_score(model, x_train, y_train, cv=kf)
     score = scores.mean()
     nni.report_final_result(score)

if __name__ == '__main__':
     RECEIVED_PARAMS = nni.get_next_parameter()
     PARAMS = get_default_parameters()
     PARAMS.update(RECEIVED_PARAMS)
     model = get_model(PARAMS)
     run(x_train, y_train, model)