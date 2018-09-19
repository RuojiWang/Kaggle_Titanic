#coding=utf-8
#这个例子主要是针对于如何进行特征工程的，其实神经网络不太吃特征工程
#大部分时候神经网络只需要低阶特征即可，少部分时候需要一些先验知识咯
#反倒是网络的设计更加重要，特征工程的问题解决以后就准备尝试网络的设计
#我突然很想跨界到金融领域咯，我觉得金融领域可能更有前景一些吧。
#毕竟计算机、机器学习或者神经网络只是一个工具，需要依托于具体的业务
#去阿里巴巴之类的地方做机器视觉、推荐系统这些东西实在太无聊了吧，可能金融领域有出路吧。
#工资待遇这些应该不会太差，而且应该可以遇到很多志同道合的朋友为以后的事业打下基础咯。
#我突然有个想法啊，我觉得神经网络的设计也完全可以采用超参搜索的形式进行吧？
#使用计算机的核心就是想办法将待解决的问题转化为计算机可以计算的问题，所以..就是超参搜索咯
#所以说，似乎我以后理论上可以无伤的开始刷题之旅咯？但是实战可能会遇到很多的问题吧..我有信心..
import pandas as pd
import featuretools as ft
#下面的这个例子展示了如何使用featuretools进行kaggle竞赛的过程，感觉很nice的样子
#https://www.kaggle.com/frednavruzov/auto-feature-generation-featuretools-example
#或者官网上这个例子不知道有没有什么卵用处呀https://github.com/Featuretools/featuretools
#下面这个例子从头到尾读一哈，我感觉可能对featuretools的作用比较了解了，还是吃了英语不好的亏，迄今为止我觉得这个有卵用，能欧提升效率
#https://github.com/WillKoehrsen/automated-feature-engineering/blob/master/walk_through/Automated_Feature_Engineering.ipynb
#这个文档阅读下来整体感觉还是有点意思比较有卵用的，但是它可能会导致一个新的问题：特征过多，还需要一些算法去解决特征过多的问题呢。
#这个网址就是上面的英文文献的中文版本https://www.jiqizhixin.com/articles/2018-06-21-2，我感觉还是有用但是可能自己用还需要核实一些细节问题
#其实我之前还有想过将其应用在Titanic的数据集上面，但是仅仅只能够使用transaction感觉意义就不是很大了
data = ft.demo.load_mock_customer()
customers_df = data["customers"]

sessions_df = data["sessions"]
print(sessions_df.sample(5))
print()

transactions_df = data["transactions"]
print(transactions_df.sample(5))
print()

entities = {"customers" : (customers_df, "customer_id"),
            "sessions" : (sessions_df, "session_id", "session_start"),
            "transactions" : (transactions_df, "transaction_id", "transaction_time")
            }

relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]

feature_matrix_customers, features_defs = ft.dfs(entities=entities, 
                                                 relationships=relationships,
                                                 target_entity="customers")

feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="sessions")

print(feature_matrix_sessions.head(5))
print()

data_df = pd.DataFrame.from_dict(data, orient='index')
data_df.to_csv("C:/Users/1/Desktop/data.csv")
customers_df.to_csv("C:/Users/1/Desktop/customers_df.csv")
sessions_df.to_csv("C:/Users/1/Desktop/sessions_df.csv")
transactions_df.to_csv("C:/Users/1/Desktop/stransactions_df.csv")
feature_matrix_sessions.to_csv("C:/Users/1/Desktop/feature_matrix_sessions.csv")

features_defs_df = pd.DataFrame({'col':features_defs})
features_defs_df.to_csv("C:/Users/1/Desktop/features_defs_df.csv")
