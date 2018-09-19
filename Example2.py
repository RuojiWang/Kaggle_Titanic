#coding=utf-8
import sklearn.datasets
import numpy as np
import random

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

paramgrid = {"kernel": ["rbf"],
             #np.logspace用于创建等比数列，从-9到9共创建25个
             "C"     : np.logspace(-9, 9, num=25, base=10),
             "gamma" : np.logspace(-9, 9, num=25, base=10)}

random.seed(1)

#现在为止，我认为遗传算法还是略胜于超参搜索的
#超参搜索仅仅能够搜索一些经验参数的组合缺乏活力！
#如果这个能够支持cuda那就很完美了啦~
#我现在感觉对于这个EvolutionaryAlgorithmSearchCV的参数非常不了解呀
#我在cv.py中观看EvolutionaryAlgorithmSearchCV的fit函数中发现了
#如何保存搜索到的最优学习器的办法，似乎不用担心与随机超参搜索结合的问题咯
#然后我还在想是不是用随机超参搜索给定经验数据，然后遗传算法再用经验数据计算
#或者是同时用随机超参搜索算法与遗传算法搜索，然后选出二者中的最贱参数咯？

#我想了一下我觉得自己对遗传算法参数看不懂的原因可能在于不了解其运行过程机制
#下面是一个通俗解释遗传算法的例子咯：
#我们先假设你是一国之王，你想让国家后代都是好人
#首先，我们设定好了国民的初始人群大小。
#然后，我们定义了一个函数，用它来区分好人和坏人。
#再次，我们选择出好人，并让他们繁殖自己的后代。
#最后，这些后代们从原来的国民中替代了部分坏人，并不断重复这一过程。
#详见如下网页：https://zhuanlan.zhihu.com/p/28328304 
#遗传算法真的是很精彩很有用的算法，我一直觉得生物仿生才是最强的，敬畏生命敬畏自然！
#遗传算法还可以用于特征选择，遗传算法还有一些著名的python库，如TPOT（树形传递优化技术）
#倘若真的可以用TPOT选择特征顺便用遗传算法搜索一下解空间，那么以后比赛就很简单咯。
#这可能就是框架四了，自动选择特征配合上框架三就无伤刷竞赛了。框架三完了就提交Titanic和DR

#阅读遗传算法原理、阅读源代码以及经过我的测试，一些关键参数的含义如下：
#params其实就是待优化的参数，随便给一点初始参数吧
#verbose按照注释里面的说法是啰嗦，控制屏幕上输出的多少
#我是说这个参数怎么会默认设置为1，设为0时每代的进化信息都不给了
#当其设置为2的时候信息过多了，还是设置为1的时候最合适。
#population_size就是族群的样本的数目，太大太小都不好吧，看样本数据咯
#gene_mutation_prob就是基因突变的概率，我也感觉太大太小都不好咯
#gene_crossover_prob这个就是基因上染色体交换的概率
#tournament_size这个参数可能是重复下述2、3、4步骤的次数咯。

#所以我们总结出遗传算法的一般步骤：
#开始循环直至找到满意的解。
#1.评估每条染色体所对应个体的适应度。
#2.遵照适应度越高，选择概率越大的原则，从种群中选择两个个体作为父方和母方。
#3.抽取父母双方的染色体，进行交叉，产生子代。
#4.对子代的染色体进行变异。
#5.重复2，3，4步骤，直到新种群的产生。

#还有程序的输出：Types [1, 2, 2] and maxint [0, 24, 24] detected
#这个Types感觉可以不用太在意，maxint应该是每个参数的总数吧，写出[1, 25, 25]可能更合理
#以及这些参数的含义：gen    nevals    avg    min    max    std
#第一个参数和后面三个参数很好理解，第三个参数可能是平均值，第二参数可能是新增
#我阅读源代码的时候看到的toolbox.register的函数想必就是回调函数吧，感觉这个可以加以利用
#从cv.fit(X, y)进入源代码一直单步调试发现sklearn-deap的源代码真多呢。
from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=20,
                                   n_jobs=1)
cv.fit(X, y)

#这个的代码在optimize.py里面和我之前常用的fit其实还不太相似
from evolutionary_search import maximize

def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
#这两个东西的返回结果都用"_"表示的吗，这也太秀了吧，下面的这个函数在注释里面如下写到
#Same as _fit in EvolutionarySearchCV but without fitting data. More similar to scipy.optimize.
best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=1)
#我勒个去，print()了这个_我才真的觉得这个东西很像_fit咯
#print(best_params)
#print(_)
#这个问题的极值刚好在这个地方咯
#verbose设置为1的时候才有输出不然根本没输出的