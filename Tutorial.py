"""
#coding=utf-8
import torch
from torch.autograd import Variable

#卧槽下面的代码说明我可以采用GPU进行计算咯
#我这个显卡是不支持GPU计算的只有英伟达的显卡才能这么玩儿的吧
print(torch.cuda.is_available()) 
x = torch.Tensor(5, 3)
print(x)
"""

"""
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
class Net(nn.Module):
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        #PC版本才会显示下面函数的含义吗，我之前家里的台式机都不显示这个Conv2d函数的含义
        #这里只有卷积和线性连接，并没有所谓的重采样的函数在里面呀，
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*5*5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
 
    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    #为什么卷积函数定义在init函数中而池化函数定义在前向过程当中呢？大概是因为卷积不能够有反向传播吧？
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x)) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x
 
    #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:] # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

 
net = Net()
net
 
# 以下代码是为了看一下我们需要训练的参数的数量
print net
params = list(net.parameters())
 
k=0
for i in params:
    l =1
    print "该层的结构："+str(list(i.size()))
    for j in i.size():
        l *= j
    print "参数和："+str(l)
    k = k+l
 
print "总参数和："+ str(k)
#这样子看起来Pytorch上手比我想象中确实要简单一些的呢。
"""

"""
#coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
 
# torchvision输出的是PILImage，值的范围是[0, 1].
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.CIFAR10(root='C:\\Users\\1\\Desktop\\cifar-10-python\\cifar-10-batches-py', train=True, download=False, transform=transform)
 
#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
 
classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(len(trainset))
print(len(trainloader))
 

#下面是代码只是为了给小伙伴们显示一个图片例子，让大家有个直觉感受。
# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
 
# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
 
# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
"""

"""
#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(), \
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
 
#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
 
#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True, num_workers=2)
 
#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
 
#将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
net = Net()
 
criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
 
for epoch in range(2): # 遍历数据集两次
     
    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0): 
        # get the inputs
        inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]
         
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
         
        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0
         
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
        loss.backward() #反向传播
        optimizer.step() #用SGD更新参数
         
        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 2000 == 1999: # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
 
print('Finished Training')
 
correct = 0
"""

"""
#coding=utf-8
import torch
from torch.autograd import Variable

#卧槽下面的代码说明我可以采用GPU进行计算咯
#我这个显卡是不支持GPU计算的只有英伟达的显卡才能这么玩儿的吧
print(torch.cuda.is_available()) 
x = torch.Tensor(5, 3)
print(x)
"""

"""
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
class Net(nn.Module):
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        #PC版本才会显示下面函数的含义吗，我之前家里的台式机都不显示这个Conv2d函数的含义
        #这里只有卷积和线性连接，并没有所谓的重采样的函数在里面呀，
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*5*5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
 
    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    #为什么卷积函数定义在init函数中而池化函数定义在前向过程当中呢？大概是因为卷积不能够有反向传播吧？
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x)) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x
 
    #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:] # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
 
net = Net()
net
 
# 以下代码是为了看一下我们需要训练的参数的数量
print net
params = list(net.parameters())
 
k=0
for i in params:
    l =1
    print "该层的结构："+str(list(i.size()))
    for j in i.size():
        l *= j
    print "参数和："+str(l)
    k = k+l
 
print "总参数和："+ str(k)
#这样子看起来Pytorch上手比我想象中确实要简单一些的呢。
"""

"""
#coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision输出的是PILImage，值的范围是[0, 1].
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
#卧了个槽，家里面的带GPU的Pytorch版本能够看到Compose函数的定义
#但是在我的另外一台笔记本上不支持GPU的版本不能够看到Compose函数
transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
#原来这个路径是这样设置的，之前路径的设置都存在问题呢。
trainset = torchvision.datasets.CIFAR10(root="C:\\Users\\win7\\Desktop\\cifar-10-python", train=True, download=False, transform=transform)

#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(len(trainset))
print(len(trainloader))


#下面是代码只是为了给小伙伴们显示一个图片例子，让大家有个直觉感受。
# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
 
# show some random training images
#下面这行代码一直报错，老子可能花了一个半小时去查找这个问题，主要刚开始没找对关键词
#这一行代码中trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
#修改为num_workers=0就可以解决这个问题咯。这个问题的原因是windows下0.4.0之前的版本的问题吧，这个示例代码是Linux下面的
dataiter = iter(trainloader)
images, labels = dataiter.next()
 
# print images
imshow(torchvision.utils.make_grid(images))
# print labels
#哇塞，终于看到这代码给出的图片了，感觉很开心的样子呢。
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
"""

#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
#我查了一下这个PIL其实是Python Image Library的意思是 Python 平台处理图片的事实标准
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
 
#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.CIFAR10(root="C:\\Users\\win7\\Desktop\\cifar-10-python", train=True, download=False, transform=transform)
 
#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
#我现在的问题是，为什么这个.util下面要加一条下划线呢。。这个暂时不知道是怎么回事儿，但是我发现了一件事情这是我之前没有想到过的。
#其实torch和utils以及data都是文件包，也就是package的形式存在的，只有最后一个dataloader是以.py文件的形式存在的，这个我也是偶然用everything搜索得到的结果
#但是还是不明白为什么utils下面被画了红线，这个package我在everything搜索torch的路径下面找到了呀。可能类似VS偶尔也会莫名的错误的对代码画出下标吧。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
 
#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
testset = torchvision.datasets.CIFAR10(root="C:\\Users\\win7\\Desktop\\cifar-10-python", train=False, download=False, transform=transform)
 
#将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
#下面的这个神经网络就和前面的神经网络完全是一回事情吧
#不对呀，仔细看其实还是有蛮多不同的细节的呢。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
net = Net()
 
#这边可以详细的定义损失函数和计算方式咯
criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
 
for epoch in range(2): # 遍历数据集两次
     
    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0): 
        # get the inputs
        inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]
         
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
         
        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0
         
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
        loss.backward() #反向传播
        optimizer.step() #用SGD更新参数
         
        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 2000 == 1999: # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
 
print('Finished Training')
 
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    #print outputs.data
    #返回输出tensor中所有元素的最大值
    _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#最后输出的结果大概是这个样子的，看来这个loss其实和准确率差很多耶
#我看到loss才1.2以为学到了很吊的东西，然而准确率才54%而已呢
#[1,  2000] loss: 2.186
#[1,  4000] loss: 1.848
#[1,  6000] loss: 1.648
#[1,  8000] loss: 1.552
#[1, 10000] loss: 1.514
#[1, 12000] loss: 1.461
#[2,  2000] loss: 1.375
#[2,  4000] loss: 1.359
#[2,  6000] loss: 1.324
#[2,  8000] loss: 1.314
#[2, 10000] loss: 1.293
#[2, 12000] loss: 1.263
#Finished Training
#Accuracy of the network on the 10000 test images: 54 %

"""
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    #print outputs.data
    _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
"""