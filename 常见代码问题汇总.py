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
                    
    8.解决hosts文件无法被修改的问题
                    打开hosts所在的路径C:\WINDOWS\system32\drivers\etc
                    然后在hosts文件上点击鼠标右键，在弹出的选项中，点击打开“属性”
                    打开hosts文件属性后，切换到“安全”选项卡，然后点击选中需要更改的当前用户名，
                    然后点击下方的“编辑”在弹出的编辑权限操作界面，先点击选中需要更高权限的账户名称，
                    比如这里需要给名称为“电脑百事网”的user用户分配修改hosts文件权限，选中用户后，
                    勾选上下方的“修改”和“写入”权限，完成后，点击右下角的“应用”就可以了
        
"""