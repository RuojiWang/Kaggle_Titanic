#coding=utf-8
#这个系列的代码主要就是用于尝试迁移学习咯，我试一下titanic的东西如何进行迁移学习
#具体的资料先看看这个再说咯 https://cloud.tencent.com/developer/article/1495081
#假如两个领域之间的区别特别的大，不可以直接采用迁移学习，因为在这种情况下效果不是很好。
#在这种情况下，推荐以上的方法，在两个相似度很低的domain之间一步步迁移过去（踩着石头过河）。
#目前俺们村儿的titanic就是属于相似性差很多的情况，如果迁移学习恐怕也是一步一步的迁移过去，裂开了。。
#以后做比赛俺需要用到 https://www.jianshu.com/p/59cdbf7439df 这里面提到的fastai咯
#这里有Pytorch自带的迁移学习相关模块：
#https://pytorch.org/docs/0.3.0/torchvision/models.html和https://www.jianshu.com/p/d04c17368922
#torch.nn.NLLLoss torch.nn.CrossEntropyLoss nn.Softmax 这三个好像经常容易出现各种错误吧
#https://www.pytorchtutorial.com/mofan-pytorch-tutorials-list/ 这个是莫凡的回归网络和分类网络的例子


#这个版本是第三个版本咯，也就是准备用fastai来搞一发这个titanic咯
#先来个素的版本，也就是快速实现一个能够运行的版本试一下fastai到底如何咯
#网上代码的基础上自己再来一个超参搜索看看结果咯,zhuyaomubiaoshigaodong lr_findruheshiyong
#如果够顺利的话还可以加上超参搜索咯
#https://www.yuque.com/apachecn/fastai-ml-dl-notes-zh/zh_dl2 这里有我之前比较想要找的fastai的课程笔记
#https://blog.csdn.net/zhouchen1998/article/details/90071837 这里还有fit_one_cycle和fit的区别


#我这几天一直有点郁闷我在想fastai有各种问题但我都不知道怎么克服。比如说文档太欠缺难以使用没别人的example code我真的不知道怎么写程序咯，
#fastai代码难以修改或者调参真不知道这些都是啥呢（不调参作tabular效果其实一般，当然做机器视觉的话迁移学习就很碉），以后用这个参加比赛还得自己准备很多东西呢太费时间了，
#而且fastai对于pytorch等有版本依赖以后难以部署在生产环境的，有的时候运行还没有现实哪个lr的图导致不知道选择那个超参。
#此外我觉得我在主业方面有点过于追求效果和目的，忽略了克服困难的难度了导致效率不高，以后这一点需要注意一下。
#我在想最适合我的是不是automl，毕竟上述滴问题都会直接给出了一个靠谱的答案咯，而且刚好搜索到一个能够实现的计算机视觉/自然语言处理/害有表格数据的库
#autogluon 咯。https://github.com/awslabs/autogluon 这里就是 autogluon的主页上面有安装的相关内容，
#https://www.linuxidc.com/Linux/2019-06/159059.htm 安装的时候会遇到gcc不存在的错误按照右侧的方式进行处理即可。


#之后的所有transferlearning都直接使用 autogluon来做嘛。不过这个库目前只支持Linux和苹果的系统，可能以后会支持Windows把，也许我花在Linux上面的时间也不算白费


import datetime
import warnings
import numpy as np
import pandas as pd

from fastai import *
from fastai.tabular import *

train_df = pd.read_csv("kaggle_titanic_files/train.csv")
test_df = pd.read_csv("kaggle_titanic_files/test.csv")

for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]
    df['Deck'] = df['Cabin'].str[0]

# find mean age for each Title across train and test data sets
all_df = pd.concat([train_df, test_df], sort=False)
mean_age_by_title = all_df.groupby('Title').mean()['Age']
# update missing ages
for df in [train_df, test_df]:
    for title, age in mean_age_by_title.iteritems():
        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age

test_df.Fare.fillna(0, inplace=True)

dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
cont_names = ['Age', 'Fare', 'SibSp', 'Parch']
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train_df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           #.split_by_idx(valid_idx=range(200,400))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())

np.random.seed(101)

learn = tabular_learner(data, layers=[600, 200], metrics=accuracy)
#learn = tabular_learner(data, layers=[600,200], metrics=accuracy, emb_drop=0.1)
#learn.fit(10)

learn.lr_find()
learn.recorder.plot()
#learn.sched.plot()
learn.fit(30, 1e-3)

# get predictions
preds, targets = learn.get_preds()
predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)

predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)

sub_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})
sub_df.to_csv('submission.csv', index=False)

sub_df.tail()
print("mother fucker~")