import torch.nn as nn
#coding=utf-8
#我的天，我刚才发愁写不出这个东西然后去床上睡了四个小时
#在睡觉的时候做了很多的梦得到了很多的神级启示惊呆了我
#比如说我梦到神经网络的一百个证明，证明了98个利用了两个公理
#我不知道是不是已经存在神经网络的一百个证明，还是神启发我去做这事儿
#如果想要做这件事情可能需要torch.nn.Sequential的帮助吧，不然感觉挺麻烦的
#Andrew Ng在ML课程中提到输入层：神经元个数=feature维度
#输出层：神经元个数=分类类别数。隐层： 默认只用一个隐层
#如果用多个隐层，则每个隐层的神经元数目都一样
#隐层神经元个数越多，分类效果越好，但计算量会增大
#我感觉在设计之前是不是在过一遍他们的之前的讲义呢？
#神经网络的普适性原理很炫酷：可以用一层隐函数拟合任何函数（足够的隐藏单元）
#李宏毅的thin all vs fat short.这个东西真的可以自动设计吗，那么多元素在内需要考虑
#nn.Conv1d 2d 3d,nn.Linear,F.relu,F.max_pool2d,
#https://ptorch.com/docs/1/torch-nn这里面其实有所有的nn相关的元素
#看了这个以后我觉得只能够使用nn.Sequential不然我不可能找出新的规范的吧
#我通过ctrl代码中Sequential的方式查到类似的还有ModuleList和ParameterList
#所以我现在似乎找到了思路，下面这个函数生成的东西经过ModuleList之后变为模型咯
#我之所以迟迟无法下手的缘故就是因为想要找到能够面面顾及的解决方案，但是并不存在吧
#这个问题的难点主要在：1）接口设计，兼容我的超参甚至是其他超参 2）模型的保存问题咯
def auto_model_creator(input_nodes, hidden_layers, hidden_nodes, output_nodes):
    
    layers_list=[]
    
    if (hidden_layers==0):
        nn.Linear(input_nodes, output_nodes)
        
    