#coding=utf-8
#这个文档写的是在太简略了吧，主要是看在这个库支持深度学习不然我根本没兴趣。
#卧槽怪不得这个文档这么简略原来只有一个处理图片的class 因为在autokeras下面只有一行
#from autokeras.classifier import ImageClassifier .classifier只有一个class ImageClassifier
#总体感觉这个库确实是比较简单的吧，而且能够找到的例子就二个都是关于图像处理相关的
#https://github.com/jhfjhfj1/autokeras/tree/master/tests 测试前面还有一些例子
#可以试一下这个能否应用于Titanic的数据集上面，我在想github上这个东西很有可能被过誉了吧
#但是有一点还是挺好的，这个东西是基于Pytorch的，总比基于tensorflow好多了吧，至少我还能试
"""
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

#其实真正的代码就在下面三行而已，前面的全部都是获取数据并预处理
#我用这个模型接入了Titanic的数据然后单步调试发现了很多有意思的东西
#至少这个贝叶斯优化器以及超参搜索的参数似乎是文档中鲜有提到的
clf = ak.ImageClassifier()
#我感觉好像单步调参挑不出问题在哪里呢，干脆还是
clf.fit(X_train_scaled, Y_train)
#results = clf.predict(x_test)
"""

#https://www.codetd.com/article/2940603
#下面终于找到了一个autokeras的代码与数据
#给的示例代码没有数据，这个做的真心垃圾吧
#迄今为止吧，我觉得我的Titanic数据可以伪装成为图片进行学习的吧？
import os
import csv
import pickle

#TRAIN_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train'
#TRAIN_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train_labels.csv'
#TEST_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test'
#TEST_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test_labels.csv'
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

#TEST_IMG_DIR_INPUT = "/home/yourname/Documents/tensorflow/images/500pics2/test_origin"
#TEST_IMG_DIR_OUTPUT = "/home/yourname/Documents/tensorflow/images/500pics2/test"
#TRAIN_IMG_DIR_INPUT = "/home/yourname/Documents/tensorflow/images/500pics2/train_origin"
#TRAIN_IMG_DIR_OUTPUT = "/home/yourname/Documents/tensorflow/images/500pics2/train"
TEST_IMG_DIR_INPUT = "C:/Users/1/Desktop/re/test"
#在win7下面这种写法是可以的，但是win10下面这种代码就无法运行
#原来是可以下面这样的方式写的，原因是因为没有没有对应的文件夹
#不过我觉得这个也太弱智了吧，居然不能够自己建立文件夹的么，居然给我报错
TEST_IMG_DIR_OUTPUT = "C:/Users/1/Desktop/re/test_scaled"
#TEST_IMG_DIR_OUTPUT = "C:\\Users\\win7\\Desktop\\re\\test_scaled"
TRAIN_IMG_DIR_INPUT = "C:/Users/1/Desktop/re/train"
TRAIN_IMG_DIR_OUTPUT = "C:/Users/1/Desktop/re/train_scaled"
#TRAIN_IMG_DIR_OUTPUT = "C:\\Users\\win7\\Desktop\\re\\train_scaled"
IMAGE_SIZE = 28

def format_img(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        path_name = os.path.join(input_dir, file_name)
        img = image.load_img(path_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        path_name = os.path.join(output_dir, file_name)
        img.save(path_name)

format_img(TEST_IMG_DIR_INPUT, TEST_IMG_DIR_OUTPUT)
format_img(TRAIN_IMG_DIR_INPUT, TRAIN_IMG_DIR_OUTPUT)


#这个可能是老版本的代码吧，居然出现了autokeras.image_supervised
#妈卖批我居然Google了一下还没有查到代码还是我在Eclipse跳转查到的位置
#from autokeras.image_supervised import load_image_dataset, ImageClassifier
from autokeras.classifier import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array

import numpy as np
import pandas as pd
#from pydev import pydevd这个写法是错误的，根本没有'pydev'这种东西呢。

#TRAIN_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train_labels.csv'
#TRAIN_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train'
#TEST_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test_labels.csv'
#TEST_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test'

TRAIN_CSV_DIR = "C:/Users/1/Desktop/re/train_labels.csv"
TRAIN_IMG_DIR = "C:/Users/1/Desktop/re/train"
TRAIN_IMG_DIR_SCALED = "C:/Users/1/Desktop/re/train_scaled"
TEST_CSV_DIR = "C:/Users/1/Desktop/re/test_labels.csv"
TEST_IMG_DIR = "C:/Users/1/Desktop/re/test"
TEST_IMG_DIR_SCALED = "C:/Users/1/Desktop/re/test_scaled"

#PREDICT_IMG_PATH = '/home/yourname/Documents/tensorflow/images/500pics2/test/719.jpg'
PREDICT_IMG_PATH = "C:/Users/1/Desktop/re/test_scaled/719.jpg"

#MODEL_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/model/my_model.h5'
#MODEL_PNG = '/home/yourname/Documents/tensorflow/images/500pics2/model/model.png'
MODEL_DIR = "C:/Users/1/Desktop/re/model/my_model.h5"
MODEL_PNG = "C:/Users/1/Desktop/re/model/model.png"
IMAGE_SIZE = 28

if __name__ == '__main__':
    # 获取本地图片，转换成numpy格式
    #下面的两行代码读取的数据在clf.fit(train_data, train_labels, time_limit=1 * 60)的时候会出现一些错误
    #我觉得很费解的就是为什么下面的两行代码会造成clf.fit(train_data,报错说含有非数字的数据呢？
    #train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=TRAIN_IMG_DIR)
    #test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=TEST_IMG_DIR)
    #下面的这种写法似乎是可以执行到clf.fit(train_data, train_labels, time_limit=1 * 60)不报错，但是好像没执行出结果
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=TRAIN_IMG_DIR_SCALED)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=TEST_IMG_DIR_SCALED)

    # 数据进行格式转换
    #to_csv写出来的数据非常的奇怪啊，我实在不知道怎么整的
    #直接执行下面的.astype('float32')居然直接报错了
    #pd.DataFrame(data=train_data).to_csv("C:/Users/1/Desktop/train_data.csv")
    #pd.DataFrame(data=test_data).to_csv("C:/Users/1/Desktop/test_data.csv")
    #难道是因为路径出问题了吗，划分出来的train_data怎么是这个尿性的呢
    #train_data应该是读取的像素点吧，怎么看都感觉没发现问题的呀？
    #如果只是除法的话，应该被注释掉也没有关系的吧？？？不能被注释否则type类型不对的吧没办法搜索
    #我仔细想了一下，大概是因为从TRAIN_IMG_DIR里面读取的数据是可能过多了吧
    #调试的时候以及写入文件的时候train_data出现了...，这大概就是非数字字符吧
    #如果是从TRAIN_IMG_DIR_SCALED中读取数据的时候，除了fit没结果其他都可以正常执行所以应该是上述的原因吧。
    #我了个飞天大草，使用TRAIN_IMG_DIR_SCALED路径并且使用下面的astype设置类型并做除法就可以了运行了。。。
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    #那我再尝试一下是否可以不用做除法呢，我觉得说不定可以的吧
    #实验表明将会一直在控制台输出ModuleNotFoundError: No module named 'pydevd'，这可真的是费解呢。
    #我好想要操了吧，似乎真的时不时的会出现，我看到一个解决方案居然是import pydevd，这尼玛有点秀吧。没有pydevd的吧。
    #train_data = train_data.astype('float32')
    #test_data = test_data.astype('float32')
    print("train data shape:", train_data.shape)

    # 使用图片识别器
    clf = ImageClassifier(verbose=True)
    # 给其训练数据和标签，训练的最长时间可以设定，假设为1分钟，autokers会不断找寻最优的网络模型
    #下面这个写法_validate终于没有报之前说的必须是数字的错误了，但报错说x_train必须至少是二维的
    #clf.fit(train_labels, train_labels, time_limit=1 * 60)
    #那应该还是说明了一点，train_data中的非数字应该不是type=什么之类的，而可能是出现的...吧
    #这个fit、final_fit以及evaluate之间到底有什么区别哦
    clf.fit(train_data, train_labels, time_limit=1 * 60)
    
    """
    # 找到最优模型后，再最后进行一次训练和验证
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)
    # 给出评估结果
    #顺便看了一下classifier里面的evaluate、fit和final_fit函数
    #我个人感觉这些函数的接口设计的都是蛮奇怪的吧，也许可能是因为我不太懂深度学习？？
    y = clf.evaluate(test_data, test_labels)
    print("evaluate:", y)

    # 给一个图片试试预测是否准确
    #这里似乎读错了数据，只需要读入SCALED的数据即可
    #但是是否存在那种需要读入原数据的情况呢，因为不同
    #输入尺寸的神经网络才能够接受不一样的输入尺寸的数据吧
    img = load_img(PREDICT_IMG_PATH)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    #因为训练数据似乎都是
    x = np.reshape(x, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print("x shape:", x.shape)

    # 最后的结果是一个numpy数组，里面是预测值4，意味着是马，说明预测准确
    #我刚才愣了一下为毛下面的不能够预测，突然想到是因为前面没有执行fit函数
    y = clf.predict(x)
    print("predict:", y)
    """

    """
    # 导出我们生成的模型
    #clf.load_searcher().load_best_model().produce_keras_model().save(MODEL_DIR)
    #上面一行的代码爆出了下面的错误，错误的原因在于NetworkX 2的改变，这种问题在所难免吧
    #AttributeError: 'Graph' object has no attribute 'produce_keras_model'
    #可能是networkx版本的问题，但是pip install networkx==1.9.1或者2.0或者2.1都无法解决该问题
    #最后老子又将networkx改回1.11的版本了，免得我之前用的代码不能够运行或者出现异常
    #我觉得我开始失去耐心了还是采用pickle的办法写入到文件中进行保存吧，库中也是这么封装的
    #最关键的是我上午使用Google了那么多的东西都没有用,
    best_model = clf.load_searcher().load_best_model()
    #下面的两条语句已经输出了对象的类型了，接下来直接翻阅对应代码就可以了
    #print(type(clf.load_searcher())) #输出结果是<class 'autokeras.search.BayesianSearcher'>
    #print(type(clf.load_searcher().load_best_model()))输出结果是<class 'autokeras.graph.Graph'>
    #原来Anaconda安装以后的代码都存在于这个目录下D:\Anaconda3\Lib\site-packages找到autokeras即可
    #这个bayesian里面的代码真的也是蛮奇怪的，我不太理解贝叶斯为啥需要计算layer的distance呢
    #文件保存的路径好像有点奇怪，可能是D:\autokeras\tmp吧，不然怎么会凭空多出这个文件夹呢。。
    #费了很大的劲还是没搞清楚这个到底是咋回事儿，可能是autokeras和tensorflow版本不匹配吧
    #而且文件似乎是乱存的，我觉得还是直接使用pickle存储吧，自己还能控制存储位置，他底层也是用pickle的
    best_model.produce_keras_model().save(MODEL_DIR)
    # 加载模型
    model = load_model(MODEL_DIR)
    # 将模型导出成可视化图片
    plot_model(model, to_file=MODEL_PNG)
    """
    

    #算了还是直接使用pickle进行封装吧
    files = open("autokeras_best_model.pickle", "wb")
    pickle.dump(clf, files)
    files.close()
    
    files = open("autokeras_best_model.pickle", "rb")
    best_model = pickle.load(files)
    files.close()
    
    
    """
    img = load_img(PREDICT_IMG_PATH)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    
    #这一小串代码证明了，我的模型确实是被正确的存储了下来
    x = np.reshape(x, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print("x shape:", x.shape)

    y = best_model.predict(x)
    print("predict:", y)
    """
    
    #将模型进行可视化的显示咯，但是不论怎么设置环境好像都没用
    #终于找到解决方案了而且写在了 常见代码问题汇总.py中
    #但是现在出现了新的问题：'ImageClassifier' object has no attribute 'layers'
    #我总觉得这个问题和之前的模型无法存储有一定的关系，说不定是因为keras版本不够高
    #我找到了一条安装指定版本的命令 pip install  keras==2.0.8
    #我个人觉得这个问题应该是keras和tensorflow的版本不匹配的问题吧
    #我现在的keras版本是2.2.2，刚才降到2.1.6了,然而还是这个问题咯
    #从报错而言绝壁是keras单方面的问题，肯定不是tensorflow方面的问题
    #乱输一个pip install  keras==2.18会反馈所有的正确可以安装的版本哈
    #研究这个问题已经几个小时了，可能有4个小时了吧，
    #plot_model(best_model, to_file=MODEL_PNG)
    print(best_model)
    print(best_model.__doc__)
    print(best_model.__module__)
    #print(best_model.__name__)
    #print(best_model.__qualname__)
    #print(best_model.__self__)
    #print(best_model.__text_signature__)
    
    #AttributeError: 'ImageClassifier' object has no attribute 'layers'
    #其他的输出都一切正常，这样子模型根本没办法输出吧
    #但是在autokeras的官方示例代码里面根本没有存储这些功能吧
    #plot_model(best_model, to_file=MODEL_PNG)
    