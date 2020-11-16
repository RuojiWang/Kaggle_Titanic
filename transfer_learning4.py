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


#之后的所有transferlearning都直接使用 autogluon来做嘛。不过这个库目前只支持Linux和苹果的系统，
#可能以后会支持Windows把，也许我花在Linux上面的时间也不算白费
#你敢相信这个autogluon的效果在原始数据集上面居然比我经过了处理的数据集上面效果更好


import pandas as pd
from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path="kaggle_titanic_files/train.csv")
test_data = task.Dataset(file_path="kaggle_titanic_files/test.csv")
predictor = task.fit(train_data=train_data, label='Survived')
y_pred = predictor.predict(test_data)

sub_csv = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
sub_csv.to_csv("kaggle_titanic_files/autogluon_submission.csv", index=False)

sub = pd.read_csv("kaggle_titanic_files/autogluon_submission.csv")
sub['Survived'] = y_pred