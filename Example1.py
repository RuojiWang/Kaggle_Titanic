#coding=utf-8
#以下是注册google cloud的过程，怪不得我信用卡没有出账单
#首先你需要有一个Google账户，然后登陆https://cloud.google.com/
#点击免费试用进行注册，填写基本信息和相关协议，账户类型选择个人，
#地址注意与信用卡账单地址一致，注册完成后信用卡账户会被预扣1美元，过会就会返还回来的。
import numpy as np
from sklearn.datasets import make_classification
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from skorch.net import NeuralNetClassifier

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedShuffleSplit

#查看了这个make_classification的函数以后，我发现这个其实分成了二分类呀
X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
#不加下面一行被注释的代码编译器会报错报到死，花了五个小时终于解决了
#mmp翻来覆去的google，并且阅读各种程序以及解决方案差点崩溃，终于成了！
#回想起来这个过程，官方提供的例子居然无法直接运行，我觉得很煞笔吧。
y = y.astype(np.longlong)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X
#卧槽下面两行代码加不加都无所谓，如果加下面两行代码可以去掉y = y.astype(np.longlong)
#添加y = y.astype(np.longlong)或者下面的两行代码其实是解决这个问题的两种方式而已。
#model = MyModule()
#model = model.float()
#哇塞真是特大喜讯device可以设置为cude，然后就可以直接支持GPU计算不用我再单独设置咯
net = NeuralNetClassifier(
    MyModule,
    #原来这里并不是GPU而是cuda呀
    #device="cuda",
    device="cpu",
    max_epochs=10,
    lr=0.1,
)
#上面都是定义模型的部分，下面才是对模型进行计算和训练的部分咯。
#下面是三种计算模型的方式，我觉得最心仪的其实是下面的gs搜索部分咯。


#下面这一行代码报错了，具体报错的内容如下，这是要求我将y的类型也修改了吗，修改了也没用呀，单步调试没啥发现
#Expected object of type torch.LongTensor but found type torch.IntTensor for argument #2 'target'
#这个调试了很久发现在fit函数内调用的partial_fit(X, y, **fit_params)，该函数内部划分了测试集合预测集
#这种强制划分了测试集和验证集的做法可能会导致神经网络和其他模型相比存在数据集不足的劣势。。
#不仅如此，这个可能也让我添加噪声存在一定的困难。。这大概就是说我可能需要新的框架方案吧。。
net.fit(X, y)
y_proba = net.predict_proba(X)
print(y_proba)
y_ = net.predict(X)
print(y_)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#sklearn中的pipeline主要带来两点好处： 
#1.直接调用fit和predict方法来对pipeline中的所有算法模型进行训练和预测。 
#2.可以结合grid search对参数进行选择
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('net', net),
])

pipe.fit(X, y)
y_proba = pipe.predict_proba(X)
#卧了个槽，下面这句居然没报错，感觉pytorch版本框架二好像有点搞头了
y_ = pipe.predict(X)

from sklearn.model_selection import GridSearchCV

params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 20],
    #下面的这个属性我似乎没有查到吧？感觉这个很细节的发现，我可真是个机灵鬼
    #最后查出来这个居然是MyModule的参数，但是这个参数为什么是这个写法呢？
    #下面的写法都是能够正确运行的：module__num_units、module_num_units
    #但是下面这些写法就会出错呢：module___num_units、module____num_units
    'module__num_units': [10, 20],
    #我估计写了一个肯定不存在的属性，如下行代码所示。我想通过报错发现模型属性集
    #果然这个才是最简单的办法解决这个问题，我自己调试了一下都没看到参数在哪里呢
    #Check the list of available parameters with `estimator.get_params().keys()
    #'mother_fucker':[10, 20, 30]
}
#下面就是输出属性的取值集合了，具体的输出结果如下面几行注释所示
#dict_keys(['module', 'criterion', 'optimizer', 'lr', 'max_epochs', 
#'batch_size', 'iterator_train', 'iterator_valid', 'dataset', 
#'train_split', 'callbacks', 'warm_start', 'verbose', 'device', 
#'history', 'initialized_', 'callbacks_', 'criterion_', 'module_', 
#'optimizer_', 'callbacks__epoch_timer', 'callbacks__train_loss', 
#'callbacks__train_loss__scoring', 'callbacks__train_loss__lower_is_better',
# 'callbacks__train_loss__on_train', 'callbacks__train_loss__name', 
#'callbacks__train_loss__target_extractor', 'callbacks__train_loss__use_caching', 
#'callbacks__valid_loss', 'callbacks__valid_loss__scoring', 
#'callbacks__valid_loss__lower_is_better', 'callbacks__valid_loss__on_train', 
#'callbacks__valid_loss__name', 'callbacks__valid_loss__target_extractor', 
#'callbacks__valid_loss__use_caching', 'callbacks__valid_acc', 
#'callbacks__valid_acc__scoring', 'callbacks__valid_acc__lower_is_better', 
#'callbacks__valid_acc__on_train', 'callbacks__valid_acc__name', 
#'callbacks__valid_acc__target_extractor', 'callbacks__valid_acc__use_caching', 
#'callbacks__print_log', 'callbacks__print_log__keys_ignored', 
#'callbacks__print_log__sink', 'callbacks__print_log__tablefmt', 'callbacks__print_log__floatfmt'])
#然而这些参数里面并没有所谓的num_units，结果这个参数居然是MyModule的参数这样我可以对结构进行超参搜索咯，很强
#print(net.get_params().keys())
gs1 = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')
gs1.fit(X, y)
#他妈的，如果要使用下面两个函数一定要设置refit=True
#我在想为什么在example中他非要设置refit为false呢？
#我查阅了net(skorch关于NeuralNetClassifier)的部分
#以及_search（sklearn关于GridSearchCV与RandomizedSearchCV）的部分
#以及我对best_estimator_的理解，我觉得实现了GS和RS应该这些就能正常调用吧？
#在经过了这些思考之后，我觉得我可能多虑了，直接设置refit=True就完事儿了
#print(gs1.score(X, y))
#gs1.predict(X)
#gs1.best_estimator_.predict(X)

#这个实验证明RandomizedSearchCV显然还是可以进行计算的呀，为毛我的框架中不能运行呢
#至于不能放到框架中，大概是因为输入的类型的问题吧，毕竟只有ndarray类型输入才行吧
#毕竟下面的代码能够正常运行就已经说明了RS是能够支持这些接口滴
gs2 = RandomizedSearchCV(net, params, refit=True, cv=3, scoring='accuracy', n_iter=4)
gs2.fit(X, y)
#果然pytorch并不支持.score属性，是因为refit被设置为False的缘故。
print(gs2.score(X, y))
print(gs2.best_score_, gs2.best_params_)