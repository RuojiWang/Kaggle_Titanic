"""
    如何在Windows下安装torchvision
        一共有三种方式进行安装，命令分别如下：
    Anaconda下面的安装方式:
        conda install torchvision -c pytorch
    pip指令的安装过程：
        pip install torchvision
    From source的安装过程:
        python setup.py install
        
    我看到单纯使用Pytorch似乎没有办法解决超参搜索的问题，那么安装一个skorch试试呢？
    skorch安装命令
        pip install -U skorch
        
        示例代码：
        import numpy as np
        from sklearn.datasets import make_classification
        import torch
        from torch import nn
        import torch.nn.functional as F

        from skorch.net import NeuralNetClassifier

        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)

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

        net = NeuralNetClassifier(
            MyModule,
            max_epochs=10,
            lr=0.1,
        )

        net.fit(X, y)
        y_proba = net.predict_proba(X)
    
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('net', net),
        ])

        pipe.fit(X, y)
        y_proba = pipe.predict_proba(X)
    
        from sklearn.model_selection import GridSearchCV

        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'module__num_units': [10, 20],
        }
        gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

        gs.fit(X, y)
        print(gs.best_score_, gs.best_params_)
        
        导入skorch失败解决方案：        
            运行程序显示ModuleNotFoundError: No module named 'skorch.net'; 'skorch' is not a package
            排查到原因是因为这个项目中存在一个叫做skorch.py文件，将文件名修改以后即可   
   
    
    如何使用cuda进行训练：
        对于普通的张量，可以直接：
        x = torch.randn(2, 3)
        x = x.cuda()
    
        对于神经网络：
        model = MyModel()
        model.cuda()
    
    报错：AttributeError: module 'torch' has no attribute 'no_grad'
        这是因为torch.no_grad只有在Pytorch 0.4以上版本才有的
            在cmd中输入指令：conda install pytorch=0.4.0 -c pytorch即可
        
    使用IPDB调试Python代码
    IPDB（Ipython Debugger），和GDB类似，是一款集成了Ipython的Python代码命令行调试工具，可以看做PDB的升级版。
        安装指令：
        IPDB以Python第三方库的形式给出，使用pip install ipdb即可轻松安装。
        通过在代码开头导入包，可以直接在代码指定位置插入断点。如下所示：
        import ipdb
        # some code
        x = 10
        ipdb.set_trace()
        y = 20
        # other code
        则程序会在执行完x = 10这条语句之后停止，展开Ipython环境，就可以自由地调试了。
        详细调试方法见：https://xmfbit.github.io/2017/08/21/debugging-with-ipdb/
  
    如何使用遗传算法进行机器学习：
        最近 在思考超参搜索以外的方式获得更优解，意外发现sklearn-deap可利用其遗传算法进行学习。
        一共有三种方式进行安装，命令分别如下：
        pip指令的安装过程：
            pip install sklearn-deap
        From source的安装过程（需cmd进入到setup.py所在路径下）:
            python setup.py install    
    Example of usage:
        import sklearn.datasets
        import numpy as np
        import random

        data = sklearn.datasets.load_digits()
        X = data["data"]
        y = data["target"]

        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold

        paramgrid = {"kernel": ["rbf"],
            "C"     : np.logspace(-9, 9, num=25, base=10),
            "gamma" : np.logspace(-9, 9, num=25, base=10)}

        random.seed(1)

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
                                   generations_number=5,
                                   n_jobs=4)
        cv.fit(X, y)

    Output:
        Types [1, 2, 2] and maxint [0, 24, 24] detected
        --- Evolve in 625 possible combinations ---
        gen    nevals    avg         min        max
        0      50        0.202404    0.10128    0.962716
        1      26        0.383083    0.10128    0.962716
        2      31        0.575214    0.155259    0.962716
        3      29        0.758308    0.105732    0.976071
        4      22        0.938086    0.158041    0.976071
        5      26        0.934201    0.155259    0.976071
        Best individual is: {'kernel': 'rbf', 'C': 31622.776601683792, 'gamma': 0.001}
        with fitness: 0.976071229827
    
    Example for maximizing just some function:
    
        from evolutionary_search import maximize
    
        def func(x, y, m=1., z=False):
        return m * (np.exp(-(x**2 + y**2)) + float(z))

        param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
        args = {'m': 1.}
        best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=False)

    Output:
        best_params = {'x': 0.0, 'y': 0.0, 'z': True}
        best_score  = 2.0
        score_results = (({'x': 1.0, 'y': -1.0, 'z': True}, 1.1353352832366128),
        ({'x': -1.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
        ({'x': 0.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
        ({'x': -1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
        ({'x': 1.0, 'y': 1.0, 'z': True}, 1.1353352832366128),
        ({'x': 0.0, 'y': 0.0, 'z': False}, 2.0),
        ({'x': -1.0, 'y': -1.0, 'z': False}, 0.36787944117144233),
        ({'x': 1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
        ({'x': -1.0, 'y': -1.0, 'z': True}, 1.3678794411714423),
        ({'x': 0.0, 'y': -1.0, 'z': False}, 1.3678794411714423),
        ({'x': 1.0, 'y': -1.0, 'z': False}, 1.1353352832366128),
        ({'x': 0.0, 'y': 0.0, 'z': True}, 2.0),
        ({'x': 0.0, 'y': -1.0, 'z': True}, 2.0))
        
    运行的时候如果遇到下述异常：
        RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
        
                经过本人Google到下述网页，立刻猜想出问题所在
        https://github.com/rsteca/sklearn-deap/blob/master/evolutionary_search/cv.py
                果然又是n_jobs这个参数的设置问题，之前Pytorch的Tutorial就出过这个问题。
        windows下最简单的解决方案就是将其设置为1，其他解决方案过于复杂并不推荐。
        
    Eclipse中无法查看某一部分源代码：
                常见的造成这种情况的原因是代码冲突，举例来说，如果同时安装了两个版本的A库就跳转到某一函数的定义之处。
    
    如何在Windows下安装advisor
    Run the advisor server.
        pip install -r ./requirements.txt
        ./manage.py runserver 0.0.0.0:8000
    pip指令的安装过程：
        pip install advisor_clients
        
        client = AdvisorClient()
    Example of usage:
        # Create the study
        study_configuration = {
            "goal": "MAXIMIZE",
            "maxTrials": 5,
            "maxParallelTrials": 1,
            "params": [
                    {
                        "parameterName": "hidden1",
                        "type": "INTEGER",
                        "minValue": 40,
                        "maxValue": 400,
                        "scallingType": "LINEAR"
                    }
            ]
        }
        study = client.create_study("Study", study_configuration)

        # Get suggested trials
        trials = client.get_suggestions(study, 3)

        # Complete the trial
        client.complete_trial(trial, trial_metrics)

    使用安装贝叶斯优化：
        贝叶斯优化的库有三种之多，因为并没有找到相关库的比较报告，我可能会挨着试一下的吧：
        半个月之后再看当时对于这两个库的选择，毫无疑问我选择hyperopt是对的，因为我现阶段不可能使用tensorflow
        所以就不太可能使用ZhuSuan了，而且hyperopt实现了sklearn这样我所有库都以sklearn为核心,会非常方便BO这个库就不行啦        
            其次是安装hyperopt：
        pip install hyperopt
            如果遇到奇怪的报错那么就在cmd中执行下列的命令：
        pip3 install networkx==1.11
        
            首先是安装Bayesian Optimization（这个个人觉得没上面的好用就不用咯）：
        pip install bayesian-optimization
        
    更新某个已经安装的库：
        再次输入其安装命令即可，例如想要将skorch更新到最新版本，则输入下列指令即可：
        pip install -U skorch
        
    安装hyperopt-sklearn:
        因为hyperopt本身还有很多的缺陷，比如说想要实现将训练好的模型用于预测居然还有自己写代码
        虽然我个人觉得肯定还是能够将这些功能的代码写的比较native，但是直接安装hysklearn能不勉强呀
            首先是windows下安装Git，因为找到的安装方式都需要用Git下载，下载之后一路next运行并将安装路径加入到环境变量即可
            然后在cmd下面进入到Git下载的上述目录，输入python setup.py install即可
    
    代码跳转的问题：
    eclipse中经常有需要跳转查看函数定义或者类定义的时候，大部分时候讲鼠标移动至对象上点击弹出的方框即可，
        但是上述方法并不是每次都能够起到效果，这个时候只需要将鼠标放置到代查看对象上，然后按键盘的ctrl键即可，
        经过安装之后，所有的函数和类都存在于本地，均支持上述的跳转，哪怕安装文件已经被删除。
        我之前还很为hyperopt中的代码无法跳转大伤脑经呢，我就说除了单步调试进入到跳转代码以外应该还有别的办法咯。
  
      如何升级pip:
    python -m pip install --upgrade pip
    
    如何安装TPOT的库：
        输入如下指令：
    conda install numpy scipy scikit-learn pandas
    pip install deap update_checker tqdm stopit
    pip install xgboost
    pip install scikit-mdr skrebate
    pip install tpot
    
    如何安装featuretools:
        输入如下指令：
    conda install -c featuretools featuretools
    
    如何安装auto keras:
    pip install autokeras
    安装autokeras失败： 
    Could not find a version that satisfies the requirement torch==0.4.0 (from autokeras)
        这个网址给出了具体的解决方案https://qiita.com/ruteshi_SI_shiteru/items/93ffd1161ff219d286fc
        最靠谱的方法是将https://github.com/hiteshn97/autokeras/ 下载下来然后使用python setup.py install安装
        虽然显示了执行上述命令的时候显示了一些奇怪的东西但是auto keras似乎确实是安装起来了呢

  Keras框架搭建:
    GPU版本
    pip install --upgrade tensorflow-gpu
    CPU版本
    pip install --upgrade tensorflow
    Keras 安装
    pip install keras -U --pre
  
  pydot和graphviz的安装
        先pip install graphviz安装graphviz
        接下来需要安装graphViz，windows环境下需要从下面的网站下载
        https://graphviz.gitlab.io/_pages/Download/Download_windows.html
        然后按照下面网站的for windows的解决方案照做就完事儿了(下载msi安装，然后同时在用户变量和系统变量下面添加东西)
        https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
            真的需要同时在用户变量（添加D:\Graphviz2.38\bin;）和系统变量下面添加（D:\Graphviz2.38\bin\dot.exe;）东西，否则无法正常使用
            但是好像直接conda install graphviz就可以了，直接conda应该不行吧，conda怎么可能添加系统变量和环境变量，这个不同时添加两个变量根本无法运行。
            通过conda安装的仅仅是调用上述库的接口而已，但是这个软件本身还是必须在windows下面安装过，否则无法运行。
        反正必须重启Eclipse否则用不了
        然后pip install pydot==1.1.0
        以上顺序一定要正确，且pydot必须是1.1.0否则出现各种奇葩的问题。（好像用了也不行啊）
        例如会遇到AttributeError: 'ImageClassifier' object has no attribute 'layers'的错误
        你会以为这是一个 tensorflow版本和keras版本不匹配的问题，升级tensorflow或者keras的版本到吐血都没卵用。
    cmd中输入conda install tensorflow==1.10.0（这是安装固定版本的语法，可以乱输一个版本反馈可以看到哪些是合法版本）无法解决，恶心死人。
        示例代码中根本没有plot_model之类的东西，大概本来这个库就不支持plot_model吧
        
    auto_ml的安装
            输入pip install auto_ml就完事儿了
            
"""