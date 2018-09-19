#coding=utf-8
import sys
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from sklearn import preprocessing
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#下面这个包是用于调试的，import他的时候没办法运行程序只能单步调试咯
#import ipdb

#这样子的写法其实不具备通用价值呀，换个问题代码都得改，无法用于skorch模型
#其实Pytorch比起skorch而言确实是要灵活一些的
#刚才看了一下这个Dataset是抽象的类，使用必须重写__len__()和__getitem()__
class PassengerDataset(Dataset):
    def __init__(self, df, test):
        #这个test其实就是记录了一个True或者False而已，并没有实质性的作用呀？
        #原来这个test字段的作用体现在了__getitem__函数内，表示是否划分tensor..
        self.test = test
        #这里面大概是这样的：[130, 496, 620, 194, 217, 588, 418, 169,...] 
        self.passergers_id = df['PassengerId'].tolist()
        #卧槽你妈，我真的是不是睿智啊？我一直再查这个函数是啥作用，结果就在下面
        #主要原因是eclipse上面的有些东西不能够显示定义的位置咯，为毛会这样呢？
        #其实这里的初始化已经进行了数据的预处理了呢，因为下面的这个函数预处理
        self.passengers_frame = self.__prep_data__(df)
        #之所以增加下面这一行无意义的代码是为了让我看到上一行代码的执行结果
        #我这里看到的self.passengers_frame单步调试的返回结果如下：
        #0    1    2         3      4         5         6    7    ..
        #0    0.0  1.0  0.0  0.396990  0.000  0.000000  0.015127  ..
        #1    0.0  1.0  0.0  0.612666  0.000  0.000000  0.014102  ..
        pass

    def __len__(self):
        size, _ = self.passengers_frame.shape
        return size

    def __getitem__(self, index):
        passenger = self.passengers_frame.iloc[index].as_matrix().tolist()
        #在这里的时候test这个布尔参数排上了用场啦，是否划分为两个tensor的意思
        if self.test:
            return torch.FloatTensor(passenger), self.passergers_id[index]
        else:
            #这个label其实就是survived的字段，用于划分属性和结果的呢
            label = [passenger[0]]
            del passenger[0]
            return torch.FloatTensor(passenger), torch.FloatTensor(label)

    @staticmethod
    def __prep_data__(df):
        passengers_frame = df
        # drop unwanted feature
        #卧槽，重大发现，下面这行的passengers_frame和上面的df居然不是共用同一对象，好像也印证了我之前的结论
        passengers_frame = passengers_frame.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
        passengers_frame[['Age']] = passengers_frame[['Age']].fillna(value=passengers_frame[['Age']].mean())
        passengers_frame[['Fare']] = passengers_frame[['Fare']].fillna(value=passengers_frame[['Fare']].mean())
        passengers_frame[['Embarked']] = passengers_frame[['Embarked']].fillna(value=passengers_frame['Embarked'].value_counts().idxmax())
        # Convert categorical  features into numeric
        # 这个可以用映射的方式实现，也可以通过loc方式修改
        # 甚至可以通过LabelEncoder, OneHotEncoder的方式修改
        passengers_frame['Sex'] = passengers_frame['Sex'].map({'female': 1, 'male': 0}).astype(int)
        # Convert Embarked to one-hot
        # 下面处理这个Embarked的方式就是OneHotEncoder方式咯
        enbarked_one_hot = pd.get_dummies(passengers_frame['Embarked'], prefix='Embarked')
        passengers_frame = passengers_frame.drop('Embarked', axis=1)
        passengers_frame = passengers_frame.join(enbarked_one_hot)
        # normalize to 0-1
        #这里有正则化的过程，Titanic3的代码根本没有正则化过程
        x = passengers_frame.values  # returns a numpy array
        # 哇哇哇，果然必须是numpy.ndarray才能够在Pytorch内运行
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        passengers_frame = pd.DataFrame(x_scaled)
        return passengers_frame

#下面这个类在整个问题中并没有使用过，这个类究竟做了什么事情？
class SinDataset(Dataset):
    def __init__(self, data_size):
        self.data_size = data_size
        #从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开,包含low但不包含high
        #这个比居然用"_"表示每次的迭代变量，看得我有点晕乎乎的，但是单步调试证明了我的判断
        self.data = [random.uniform(0,1) for _ in range(data_size)]
        #还有这个比用的Labels变量感觉非常的不合适，会让别人看到困惑的变量命名。
        self.labels = [math.sin(self.data[i]) for i in range(data_size)]

    def __len__(self):
        return self.data_size

    #这个函数就是为了能够使用[]方法而必须实现的
    def __getitem__(self, index):
        return torch.FloatTensor([self.data[index]]), torch.FloatTensor([self.labels[index]])

#就这个模型而言，存在大量可以调试的工作吧？这些东西都留给框架四了吧？
#其实框架四可以先不用考虑，因为我只有在阅读了别人的论文之后才能高质量框架开发
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        #之前我一直很想看能在这里出现的所有结构，然后我找到了一个文档可以查阅
        #https://ptorch.com/docs/1/torch-nn 这里基本所有结构都枚举了滴
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.sigmoid(x)


def load_train_data():
    df = pd.read_csv("C:\\Users\\win7\\Desktop\\train.csv")
    #我将下面的test_size参数修改为0.3，以方便和之前模型比较
    #而且我现在才知道train_test_split可以只划分为两部分呀
    #我之前一直都是划分为四部分的说呢
    train, validation = train_test_split(df, test_size=0.2)
    return train, validation


def generate_kaggle_data(model):
    test_df = pd.read_csv("C:\\Users\\win7\\Desktop\\test.csv")
    test_data_set = PassengerDataset(test_df, test=True)
    test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, num_workers=4)
    survived_col = []
    id_col = []
    for index, (test_input, test_id) in enumerate(test_data_loader):
        test_input = Variable(test_input)
        test_id = test_id.numpy()[0]
        test_output = model(test_input).data.numpy()[0][0]
        predicted = 1 if test_output > 0.5 else 0
        survived_col.append(predicted)
        id_col.append(test_id)
    df_data = {'PassengerId':id_col, 'Survived':survived_col}
    df = pd.DataFrame(data=df_data)
    df.to_csv('data/test_output.csv', index=False)

#虽然 pytorch 命令式编程，声称容易 debug，可是 代码给出的错误提示可是令人相当头疼
#比如说在for epoch in range(epochs)循环中，我就找不到为什么不能单步调试的原因
def main():
    ##我试一下这个SinDataset类呢？
    #sin_dataset = SinDataset(10)
    #for i in range(0, 10):
    #    print(sin_dataset[i])
    # setup data
    train_set, validation_set = load_train_data()
    #这里仅仅执行了__init()__函数而已呢，其实下面也仅仅执行了__init()__而已
    #其实想想也是呀，肯定只能够执行init初始化呀，不然执行其他方法吗？
    train_data_set = PassengerDataset(train_set, test=False)
    #因为我发现执行了上一行代码之后train_data_set[0]就返回如下结果
    #(tensor([ 1.0000,  0.0000,  0.2083,  0.0000,  0.0000,  
    #0.0139,  0.0000,  0.0000,  1.0000]), tensor([ 0.]))
    #我他妈返反复阅读PassengerDataset类的代码，并且单步调试但没找到原因
    #我他妈恍然大悟应该是__getitem__(self, index)函数实现的取数据形成tensor
    #所以准备执行下面一行代码，以强制执行断点，进而进入__getitem__函数内部调试
    #print(train_data_set[0])
    #dataset，这个就是PyTorch已有的数据读取接口（比如torchvision.datasets.ImageFolder）
    #或者自定义的数据接口的输出，该输出要么是torch.utils.data.Dataset类的对象，
    #要么是继承自torch.utils.data.Dataset类的自定义类的对象。怪不得这个比要实现一个类继承Dataset
    #这样的话，每次我做比赛的时候都需要输入数据继承自Dataset类咯，干脆做成一个通用类型的输入类吧。
    #奇怪咯，为什么这里的num_workers=4居然没有报错，但是官方文档的num_workers=2居然报错了
    train_data_loader = DataLoader(train_data_set, batch_size=32, shuffle=True, num_workers=4)
    validation_data_set = PassengerDataset(validation_set, test=False)
    validation_data_loader = DataLoader(validation_data_set, batch_size=1, shuffle=True, num_workers=4)
    # setup network
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    # train
    #用于调试的时候设置epochs = 1，以方便观察结果
    #用于训练的时候毫无疑问要设置epochs = 100
    epochs = 100
    #为什么for循环内的东西无法单步调试呢？可以运行但无法调试也太抽象了吧？
    #反馈出来的问题是ModuleNotFoundError: No module named 'pydevd'
    #我在网上搜到的解决方案是添加下面的代码，但是无论我写下下面的代码还是将该路径放到环境变量中似乎都没有起到作用呀？
    #sys.path.append("D:\\eclipse-SDK-4.5-win32\\eclipse\\plugins\\org.python.pydev_4.5.1.201601132212\\pysrc\\pydevd.py")
    #temp = list()
    for epoch in range(epochs):
        #Sets the module in training mode.
        #This has any effect only on modules such as Dropout or BatchNorm.
        model.train()
        #呵呵，这个还有点神奇，运行到这里就直接停止了，但这样进入循环还是有问题。。
        #https://xmfbit.github.io/2017/08/21/debugging-with-ipdb/
        #ipdb.set_trace()
        #根据我的翻阅了这么多的资料，意外的发现这个可能是无法记录中间变量吧
        #因为中间变量在完成自己的使命之后就被释放了以节约空间，是说怎么没办法
        #想要记录中间变量似乎需要使用hook机制，其实就是用变量记录中间结果以免被释放
        #所谓的hook机制就是在不改变业务框架的基础上添加新功能的这么一种机制
        for index, (inputs, targets) in enumerate(train_data_loader):
            #train_data_loader之所以能够划分为index和 (inputs, targets)
            #这是因为enumerate函数能够划分为序号和内容，至于内容为何如此划分
            #因为train_data_loader = DataLoader(train_data_set,...中
            #但是train_data_set= PassengerDataset(train_set,...中
            #其实可以单步调试发现train_data_set[0]变量的形式已经是：
            #(tensor([ 1.0000,  0.0000,  0.2525,  0.0000,  0.0000,
            #0.0154,  0.0000, 0.0000,  1.0000]), tensor([ 0.]))
            #值得注意的是：只有在for epoch in range(epochs)之前才可以单步调试
            #我反复阅读PassengerDataset类的代码，发现是其__getitem__(self, index)函数所为
            
            #这个tensor是什么类型的和损失函数的类型有一定的关系NLLLoss必须改为LongTensor
            #NLLLoss’s target should be a torch.LongTensor. See here for more details: 
            #http://pytorch.org/docs/master/nn.html?highlight=nllloss#torch.nn.NLLLoss 618
            #或者参考下面网页的做法：
            #https://discuss.pytorch.org/t/problems-with-weight-array-of-floattensor-type-in-loss-function/381
            inputs, targets = Variable(inputs), Variable(targets)
            
            #现在已知的是仅能通过这种方式将中间结果保存下来
            #temp.append(index)
            #temp.append(inputs)
            #temp.append(targets)
            #全部存下来感觉很乱，直接输出中间结果试试呢
            #直接print出来感觉代码结构清晰多了
            #由于load_train_data()函数test_size=0.2
            #所以一个epoch是32*23=736≈712.8=891*0.8 
            #print("index:")
            #print(index)
            #print("inputs:")
            #print(inputs)
            #print("targets:")
            #print(targets)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # get accuracy
        correct_num = 0
        for index, (inputs, targets) in enumerate(validation_data_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs).data.numpy()
            predicted = np.where(outputs > 0.5, 1, 0)
            answer = targets.data.numpy()
            correct_num += (predicted == answer).sum()
        print('Epoch [%d/%d], Loss:%.4f, Accuracy:%.4f' % (epoch+1, epochs, loss.data[0], correct_num/len(validation_set)))
    
        #print(temp)
        #print()
    # generate_kaggle_data(model)


#卧槽，我只是试了一下这个能否在笔记本环境运行
#意外发现笔记本比我台式机慢很多，计算速度可能只有一半
#我真的日了你的吗，我不知道为什么会突然出现这个问题
#这个EOFError查看好像可以用这个try except的方式解决
#最后发现这些奇怪问题的原因可能是因为使用了ipdb的缘故吧
if __name__ == '__main__':    
    try:
        main()
    except EOFError: #捕获异常EOFError 后返回None
        pass


