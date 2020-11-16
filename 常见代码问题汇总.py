"""
    大概已经三次遇到曾经遇到过的错误提示，但是想了很久没想起来原来是怎么解决这个错误提示，
    以至于后来花了很多时间又重新摸索了一遍，所以建立这个文件将常见的需要想一下的错误记录在案
    1.代码无法调试或跳转相关问题：
        1）反馈ModuleNotFoundError: No module named 'pydevd'
        2）Eclipse中函数无法跳转到定义处：可能安装了多个版本的库（如skorch）均有这个函数
        3）安装了多个版本的skorch也导致之前的代码无法调试，卸载其中一个即可
        4）tensorflow或者底层基于tensorflow的库或者代码似乎无法进行单步调试

    2.路径错误相关问题：
        1）可能是文件路径上的文件夹不存在，文件可以存在或者不存在，但是文件夹一定要存在
        2）在windows路径表示中，那么使用\\(两个斜杠)，要么使用一个/（反斜杠）

    3.显示不存在模块、属性或者运行异常的问题：
        1）显示不存在某个模块如No module named 'skorch.net'; 'skorch' is not a package，因为我自己创建了一个shorch.py的文件
        2）可能是因为版本问题，也就是说你安装的版本和别人的版本不一致，导致不存在某些函数模块等
        3）可能是因为windows多线程运行必须放在__main___下面，建议不适用windows下多线程

    4.dataframe或者ndarray中出现非数字字符
        1）过大矩阵的时候ndarray可能会出现非数字字符“...”，可能会影响部分操作吧

    5.很奇怪无脑的报错：
        1）可能是networkx版本的问题，之前使用hyperopt好像也遇到过的吧
        pip install networkx==1.9.1将其安装到1.9.1版本，之前我使用的是pip3 install networkx==1.11
                    类似这种的语法pip3 install --upgrade tensorflow可以进行版本的升级
        2）奇怪的错误可能是因为库的版本和该库用的库的版本可能不匹配，所以出现各种函数不存在的问题

    6.使用git下载代码或者上传代码
                    参见https://www.jianshu.com/p/303ffab6b0e4
                    参见https://www.cnblogs.com/renkangke/archive/2013/05/31/conquerAndroid.html

                    下载代码命令 ：
            git clone xxxx(项目地址)
                    上传代码命令：
            git init
            git add .
            git commit -m "mmmm(注释)"
            git remote add orgin xxxx(项目地址)
            git push -u origin master

                    如果最后一句报错error: failed to push some refs to那么请按如下操作：
            git push -f（删除远程代码，将本地代码上传到远程）
                    或者
            git pull

    7.解决github上传或者下载超慢的问题
                    参见https://blog.csdn.net/Adam_allen/article/details/78997709
        windows下用文本编辑器打开hosts文件，位于C:\Windows\System32\drivers\etc目录下
                    打开 http://tool.chinaz.com/dns, 这是一个查询域名映射关系的工具
                    查询 github.global.ssl.fastly.net 和 assets-cdn.github.com 两个地址
                    多查几次，选择一个稳定，延迟较低的 ip 按如下方式添加到host文件
                    保存文件，重新打开浏览器，起飞。
        其实直接在浏览器中安装一些插件也可以实现这个目的，搜索油猴搜索github就能够找到

    8.解决hosts文件无法被修改的问题
                    打开hosts所在的路径C:\WINDOWS\system32\drivers\etc
                    然后在hosts文件上点击鼠标右键，在弹出的选项中，点击打开“属性”
                    打开hosts文件属性后，切换到“安全”选项卡，然后点击选中需要更改的当前用户名，
                    然后点击下方的“编辑”在弹出的编辑权限操作界面，先点击选中需要更高权限的账户名称，
                    比如这里需要给名称为“电脑百事网”的user用户分配修改hosts文件权限，选中用户后，
                    勾选上下方的“修改”和“写入”权限，完成后，点击右下角的“应用”就可以了

    9.解决Python控制台中输出很多warnning的问题
        import warnings
        warnings.filterwarnings('ignore')

    10.使用TPOT等产生下列错误：ImportWarning: Falling back to the python version of hypervolume module.
        Expect this to be very slow."module. Expect this to be very slow.", ImportWarning)或者
        ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
        return f(*args, **kwds)错误

                这个主要是因为你deap的版本问题，我之前自己安装过其他版本的deap
                先 pip uninstall deap
                然后 pip install deap=1.0.2

    11.使用TPOT产生下列错误：Check failed: (n) > (0) colsample_bytree=1 is too small that no feature can be included
                这个主要是因为TPOT采用了老版本的xgboost的缘故，更具体地说好像是因为老版本的xgboost使用了多线程？反正是某些设置导致的。
                解决的方法就是直接安装新版本的xgboost就完事了，md这是我查询了一个班小时论坛得到的解决方案。
                首先输入pip install xgboost==1.,由于没有这个版本，所以对话框会弹出所有版本，这个时候你可以看到最佳的版本是0.81
                接下来就直接输入pip install xgboost==0.81这一条指令就可以卸载老版本并安装新版本咯，接下来的实验就可以正常进行啦。

    12.之前遇到的很多的版本的问题，其实或许可以通过Anaconda进行一定程度的减少吧。
                比如说上面的deap版本等问题也许直接通过conda命令安装就会顺带升级的，conda的机制决定了他会顺带升级相关依赖的版本。
                但是之所以没有一直使用conda指令主要缘故在于其指令不太简洁直观，pip指令直接就是pip install 包名，只能够试一下
        conda install 包名，但是大部分时候这个指令是不能够成功执行滴。所以我的解决方案就是：使用pip安装，出问题的时候在conda安装。。

    13.如果pytorch进行训练时第一个epoch非常的慢可能的原因在于pytorch的版本过于老旧
        conda install pytorch torchvision cuda80 -c soumith 解决方案就是左边这条语句咯。

    14.电脑上面直接执行conda install -c fastai fastai 企图安装fastai的时候就出现问题了，
    貌似是能够找到合适的版本但是无法下载呢，但是网络问题很多东西下载不下来呢。然后就出现了这些错误：
    CondaHTTPError: HTTP 000 CONNECTION FAILED，就是没下载成功嘛可能有些需要翻墙或者国内的镜像才行熬
    目前就是pandas-1.1.1 mkl-2020.2 numpy-base-1.19.1 scipy-1.5.2 icc_rt-2019.0.0 python-3.6.8 这六个包没有装上
    https://cloud.tencent.com/developer/article/1572996 直接按照左侧的操作一下再输入conda install -c fastai fastai==1.0.25
    现在下载速度直接飞起，估计是清华直接把镜像搬运过来了惹。一定是在Anaconda的Prompt里面输入命令哦不然会出现其他问题。

    15.现在执行fastai的示例代码遇到的新的错误是 RemoveError: ‘setuptools’ is a dependency of conda and cannot be removed from conda’s operating environment.
    具体的解决方案是这个照着操作一遍就好了 https://blog.csdn.net/qq_42209978/article/details/101784732 但是下载速度非常滴慢不知道要弄到什么时候熬果然这个解决方案失败了
    ，但是好歹fastai能用了。阿这，期是不能够用因为一使用就报错了感觉很恼火，真的是心态都炸了。

    16.执行python setup.py install 时候遇到的问题（https://blog.csdn.net/YPP0229/article/details/106216631）
    （1）先下载你要安装的包，并解压到磁盘下；
    （2）从Anaconda Prompt激活你想要安装包的虚拟环境；
    （3）在该虚拟环境中，cd进入到刚才解压的包的setup.py 目录下
    （4）先执行 python setup.py build
    （5）然后执行 python setup.py install
    （6）可能遇到的error：UnicodeDecodeError: ‘gbk’ codec can’t decode byte 0xa2 in position 905: illegal multibyte sequence
    （7）解决方案：打开 setup.py 文件，看到：
        with open("README.md", "r") as fh: long_description = fh.read()
        改为：
        with open("README.md", "r", encoding="utf-8") as fh: long_description = fh.read()
        问题就解决啦！

    17.这个才是winows7上面anaconda安装pytorch的正解，真的司马熬浪费了我好多时间
    https://blog.csdn.net/flying_ant2018/article/details/105049006
    一定是在Anaconda的Prompt里面输入命令哦不然会出现其他问题。

    不对不对。。运行的时候会出现ModuleNotFoundError: No module named 'torch._C'。。

    18.最后找到的解决方案是直接conda安装fastai，fastai在安装的时候会自动安装pytorch和torchvision，
    只是说看如何选择支持GPU的版本（目前还不知道）。还有安装完了之后运行会有一些小的错误，Google起来解决也简单
    最后就是之前的几个包的安装，skorch（用pip安装吧不然慢的飞起），hyperopt（这个conda直接安装就行了）

    ================================================================================================
    ================================================================================================
    难受阿已经弄了几天了还没弄好。。最后似乎好像终于快弄出来了，至少接近尾声了吧,弄了三天还是四天时间真的裂开了

    现在总结一下呢：
    Anaconda安装：anaconda版本的选择其实对应了python的版本，python的版本又对应了能够安装的包，尽量不要选择太高版本的
    anaconda不然之后不好安装包，这就是为什么同样一条指令或者包在某些版本的anaconda安装的飞起另外版本的就直接失败了。
    anaconda最好配上清华的镜像这个貌似是国内安装最快的方式咯。值得一提的是不同版本的conda安装同一个包的指令略有区别，
    至于具体指令是什么这个主要要在github上面查看别人的安装指令咯。我个人是比较推荐使用python3.6.5对应的anaconda的，
    因为我现在的pipeline基本是在这个版本是建立的。

    fastai安装：一定要通过conda安装，因为他依赖的包特别多不用conda可能容易出问题，直接通过
    conda install -c fastai -c pytorch -c anaconda fastai gh anaconda（目前默认安装的是2.0.16版本）就可以安装起这个，这个貌似是没有GPU版本只说的
    然后在如果通过fastai安装的话也会自动安装对应的pytorch和torchvision（torchvision版本是0.7.0，pytorch的版本是1.6.0），目前我已经试过了是可以使用GPU的

    注意1：必须这样子安装 conda install -c fastai fastai==1.0.25才行，因为网上的代码和例子基本都是这个版本，否则会出问题，累死我惹
    注意2：然后可以执行fastai的代码了但是马上会遇到下面的错误：
      ...
      File "D:\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py", line 737, in __init__
        w.start()
      File "D:\Anaconda3\lib\multiprocessing\process.py", line 105, in start
        self._popen = self._Popen(self)
      File "D:\Anaconda3\lib\multiprocessing\context.py", line 223, in _Popen
        return _default_context.get_context().Process._Popen(process_obj)
      File "D:\Anaconda3\lib\multiprocessing\context.py", line 322, in _Popen
        return Popen(process_obj)
      File "D:\Anaconda3\lib\multiprocessing\popen_spawn_win32.py", line 65, in __init__
        reduction.dump(process_obj, to_child)
      File "D:\Anaconda3\lib\multiprocessing\reduction.py", line 60, in dump
        ForkingPickler(file, protocol).dump(obj)
      BrokenPipeError: [Errno 32] Broken pipe
    D:\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py
    然后我已经将dataloader.py备份到原文件夹下面，改名叫做dataloader（原）.py，这个文件就在上述路径
    之所以要备份是因为我看到别人的解决方案是修改num_worker为0，否则在windows下面会报错
    但是fastai的代码又没有直接修改num_workers的接口，所以说可能只能修改源代码了
    咳咳，但是根据栈调用其实问题貌似最先是出在basic_data.py上面的，所以或许需要备份basic_data.py
    但是有一点是肯定的，问题是出在DataLoader，看在哪里修改原代码可以直接解决这个问题咯，感觉就是修改basic_data.py
    basic_data.py所在的路径就是D:\Anaconda3\pkgs\fastai-1.0.25-py_1\site-packages\fastai
    在basic_data.py的DeviceDataLoader()类上打上断点，找到了创建DataLoader的位置修改初始化的num_workers就好
    默认情况下num_workers就是cpu的数目，我现在把这个文件使用的num_workers修改为0
    然而做了上述之后还是无法正常运行呢。。看来还是在VirualBox下面创建一个Ubuntu的虚拟机再配置环境吧
    windows下面的话，最终的解决方案貌似是这个https://xbuba.com/questions/56565823，就是把Pytorch降级到1.0.0的版本我试一下呢。。
    https://pytorch.org/get-started/previous-versions/ 这个网页上面是一些老版本的pytorch安装指令，我试了一下很多都还是可以运行，总比找到的绝大多数不能运行的好。
    conda install pytorch==1.0.1 torchvision==0.2.2 -c pytorch这个版本安装成功了，但是运行还是出现[Errno 32] Broken pipe 所以失败了，
    conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch这个版本安装成功了，但是运行还是出现[Errno 32] Broken pipe 所以失败了
    所以最后还是用Ubuntu吧。。或者用fastai v2据说好像可以支持Windows是吧

    强烈建议使用Ubuntu18.04的版本，这个版本比Ubuntu16.04真的好用太多了。。据说fastai在windows上面并不怎么稳定还是用Ubuntu学习fastai吧
    这个版本的fastai对应的是pytorch的1.6.0咯，因为VirualBox上面无法使用N卡的GPU，所以只能安装CPU版本的Pytorch，这个需要注意一下
    由于清华镜像源没有CPU版本的Pytorch和torchvision,最后我是离线下载pytorch-1.6.0-py3.6_cpu_0.tar.bz2和torchvision-0.7.0-py36_cpu.tar.bz2
    然后通过conda指令安装的，真的好恶心阿今天一天的时间就做了这个 https://blog.csdn.net/Suan2014/article/details/80410144（离线安装）
    这个是下载的镜像源咯https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/

    hyperopt安装：这个好像无法通过conda安装，毕竟他的github上面都没查到conda安装指令，并且直接
    用最常见的conda install hyperopt无法安装（可能是anaconda和python版本的问题，有些版本好像直
    接用上述指令是能够直接安装的，我之前装其他版本的时候貌似确实可以）。
    pip install hyperopt安装的版本是 0.2.4

    skorch安装：安装指令是pip install skorch==0.7.0 或者check_scoring conda install -c conda-forge
    skorch==0.7.0(只有安装这个0.7.0版本才和sklearn0.19.1没有冲突，我把所有版本从高到低挨着挨着
    试出来的)，这个skorch通过pip安装和conda安装似乎区别不大

    tpot安装：最后在anaconda prompt里面安装就行了 pip install tpot==0.9, conda install tpot似乎无效果耶
    而且其他版本似乎也安装不起来呢

    如果出现了：The NVIDIA driver on your system is too old (found version 9010).
    Please update your GPU driver by downloading and installing a new，说明CUDA和Pytorch的版本不匹配，建议直接升级驱动吧
    360安全卫士 功能大全里面有个360驱动大师，可以选择升级显卡的版本，选择一个不是特别新的版本以免出现不稳定的情况

    真的是血泪的教训呀。。
    ================================================================================================
    ================================================================================================
    总结一下安装包的内容：
    （1）使用conda可以将不同的环境隔离起来以避免base环境受到影响
    （2）在github上面查找相关第一首信息，大部分时候上面的安装信息是最有效的
    （3）尽量使用conda安装其次使用pip安装，可以修改.condarc文件配置镜像源，大部分时候直接pip安装都不太行。
    （4）也可以下载离线安装包然后通过conda进行安装，Linux下面我试过成功了但是Windows下面有点问题。可能是我那里设置出现问题了吧。
    （5）还有一种很靠谱的方式就是在Google里面输入“包名 国内 镜像源 下载”，之前要是早点用这个方式可能安装包的时候没有那么坎坷咯
    ================================================================================================
    ================================================================================================
    决定了，以后我用automl解决比赛的问题咯，现在的问题就是不知道哪个autmoml比较屌，
    但是没关系我打算安装三个最牛批的automl，以后比赛就是在这三个automl里面反复横跳咯
    可能最牛批的就是Google的automl吧，但是是付费的我妹有办法使用熬，只有用下面三个automl咯
    需要注意一下，这三个虚拟环境之间可能会相互影响吧，所以最好是独立安装虚拟环境以后也好独立升级咯，以免影响我的base环境
    可能两三年以后我会换新的机器和环境吧，其实现在环境也没有那么复杂，反正到时候分成多个环境肯定不用担心这些的
    conda env list 可以查看已经安装好的conda环境

    autogluon安装：
    先执行conda create --name autogluon python==3.6.5 创建新的虚拟环境，activate autogluon 激活这个虚拟环境，这个包只能装在Linux上所以在虚拟机上面咯
    我在想最适合我的是不是automl，毕竟上述滴问题都会直接给出了一个靠谱的答案咯，而且刚好搜索到一个能够实现的
    计算机视觉/自然语言处理/害有表格数据的库也就是autogluon 咯。https://github.com/awslabs/autogluon 这里就是 autogluon的主页
    上面有安装的相关内容，不过只能够在Linux下面安装呢。https://www.linuxidc.com/Linux/2019-06/159059.htm 安装的时候会遇到gcc不存在的错误按照右侧的方式进行处理即可。
    总体而言安装很顺利没出啥幺蛾子

    Autokeras安装:
    先执行conda create --name autokeras python==3.6.5 创建新的虚拟环境，activate autokeras 激活这个虚拟环境
    我顺便解释一下：安装主要基于https://blog.csdn.net/weixin_43887661/article/details/106756373这篇博客，然后再此基础上做了一些调整不然我装不起。。
    （1）这篇文章所有的安装环境是基于Python 3.6.5的，好在我一直就用的这个版本，所以不太需要再重新创建一个虚拟环境，不过重新创建虚拟环境还是可以学习一下的，以后可能用得着
    （2）conda install tensorflow==2.1.0 这个就是安装GPU版本的tensorflow（我是在Github上面的安装部分查到的，这个和一些博客的写法有出入，所以一切以Github为准咯，后来发现是CPU哈）
    pip install tensorflow==2.1.0会失败的，主要原因是anaconda其实加入了镜像源所以下载很快。
    （3）pip install Keras==2.3.1 或者 conda install Keras==2.3.1。这个不指定版本就会出错，因为Keras和tensorflow以及后面的autokeras之间版本有要求
    （4）pip install autokeras==1.0.1（conda install autokeras==1.0.1会安装失败）这个不指定版本就会出错，因为Keras和tensorflow以及后面的autokeras之间版本有要求。
    安装过程中会断开几次需要多执行几次pip install autokeras==1.0.1因为安装过程中会断开

    （5）安装过程中会多次断开，但是没关系我们继续执行上次的指令（主要是pip install autokeras==1.0.1，镜像源的问题导致的下载慢）。
    可以留意一下pip安装的时候是哪个包没有安装成功，然后用anaconda安装一下这个包，因为anaconda设置了镜像源的缘故，至少安装autokeras的时候木问题

    呜呜呜，装了一天终于弄好了好激动呀。
    另外补充一个小技巧 anaconda在安装的时候可以 按下 ctrl+c打断安装。
    https://jingyan.baidu.com/article/93f9803fdd922ba0e46f5584.html 左边是切换Pycharm里面的anaconda环境的办法咯

    (6)现在代码是可以使用了，但是无法使用final_fit寻找到最优的模型，估计是autokeras的版本的问题，之前不下载tensorflow2.3.1最多用到autokeras1.0.2现在可以1.0.9咯
    https://pypi.org/project/tensorflow/#files 从这上面也就是官网直接下载 tensorflow2.3.1吧
    因为之前pip安装的时候都是下载的这个版本所以用这个版本没有问题的吧 文件名：tensorflow-2.3.1-cp36-cp36m-win_amd64.whl (342.5 MB)

    还好用了conda的虚拟环境，之前的环境坏掉了用conda activate base切换环境 再用conda remove -n autokeras --all 把之前建立的autokeras虚拟环境删除了

    (*)我查了源代码了 autokeras 1.0.9也没有final_fit函数咯，我要是早知道这个我就不再下面安装什么tensoflow2.3.1的CPU版本和GPU版本咯，真滴追求完美导致我浪费时间嘛
    后面的操作不用再进行了，直接用上的的1.0.2的 autokeras就可以咯
    如果遇到奇怪的这个错误：Python Error: No module named astunparse
    conda install astunparse

    如果遇到奇怪的这个错误：AttributeError: 'tuple' object has no attribute 'shape'，解决方案是这个
    pip install git+https://github.com/keras-team/autokeras.git@master
    pip install git+https://github.com/keras-team/keras-tuner.git@master
    pip install tensorflow==2.2.0

    sklearn.externals.joblib不存在：conda install joblib https://blog.csdn.net/qq_34769162/article/details/106961429

    ImportError: No module named 'tabulate'：pip3 install tabulate https://stackoverflow.com/questions/31757552/trouble-importing-tabulate-in-python-3-4

    我之前还想用tensorflow的离线版安装结果安装过程中还需要下载别的包就失败了，最后还是按照这篇博客https://blog.csdn.net/xd_wjc/article/details/82875108
    的办法找到合适的镜像源解决的这个问题咯。镜像源安装的时候还是遇到了中途要下载包的问题，还是直接 ctrl+c打断安装 用conda的方式安装才能够解决问题咯。
    autokeras好像只是依赖tensorflow并不依赖tensorflow-gpu也就是说这个东西并不能够在GPU上面使用吧

    （7）好不容易安装完了运行就遇到到了UserWarning: h5py is running against HDF5 1.10.5 when it was built against 1.10.4, this may cause problems
    然后期间还不停的弹出这个Python3.6.5已经停止运行的窗口，用conda安装其他包也提示权限什么的问题。原来是因为tensorflow-gpu安装的问题
    用左侧 pip3 install --user -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade tensorflow-gpu 上述三五个问题就都解决了

    （8）执行这个 pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc0#egg=keras-tuner-1.0.2rc0 否则报错
    pip install autokeras==1.0.4

    I安装:
    Microsoft NNI的安装就很顺利咯，具体参考右侧的链接https://zhuanlan.zhihu.com/p/92812335
    也就是直接用命令pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nni 就安装好了，比起autokeras真的太节省力气了。
"""