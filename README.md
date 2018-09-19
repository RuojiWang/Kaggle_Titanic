# Kaggle_Titanic

* Kaggle_Titanic is a semi-automatic deeping learning experiment code for kaggle titanic competition.  
* Kaggle_Titanic treats network structure design and network initialization and so on as hyperparameters selection. And it supports deep learning model which allows you to achieve better accuracy than traditional model like xgboost easily. 
* Semi-automatic means autimaic hyperparameters optimization including compute and save the best model in history and output prediction file and intermediate computation file, which can save you lots of time. 
* Besides determining network structure, you only work is little feature engineering like feature scale and little code change like file path or hyperparameters selection. 
* Just enjoy deep learning and build your solution based on Kaggle_Titanic for other competitions.
<br></br>

### Environment Configuration

* Anaconda pytorch skorch hyperopt and python 3.6 are mainly needed. 
* You can use the following commands: 
  * pip install anaconda
  * pip install pytorch
  * pip install -u skorch
  * pip install hyperopt
<br></br>

### Running on Titanic dataset

* Firstly, configure the running environment.
* Secondly, run examples of pytorch skorch and hyperopt.
* Thirdly, include all the files into your project and copy the files in kaggle_titanic_files which are the raw data for kaggle titanic competiton, and then replace the pd.read_csv path with your copy path. 
* Finally, you can run and debug the end_of_the_titanic6.py or save_intermediate_model1.py to learn more about it. The first file named end_of_the_titanic6.py is mainly used for network structure and other hyperparameters selection, while the other is used for saving former model or network structure including the best model.
<br></br>

### Using for Other Competition
  
* Firstly, configure the running environment.
* Secondly, run the code on the titanic dataset to learn details.
* Thirdly, determine network structure do little feature engineering and hyperparameters selection for new competiton. Which means space space_nodes best_nodes and parse_space funciton in the code are mainly needed to modified. **Remeber to modify space space_nodes best_nodes and parse_space function at the same time!**
* Finally, change the network structure and run the code until you get satisfying results. Whenever you train the model, the best hyperparameters in history and prediction file of the model will be saved. **Do not forget to save the network class definition, especially the class defination of the best model, without which you can not use or load the best model from .pickle file. I strongly recommend you to create files like save_intermediate_model1.py to save versions of selection of model structure and other hyperparameters, which benefits you in using and saving best model of different versions, and also in optimizing the  design of the network structure and the selection of other hyperparameters.** 
<br></br>

### Contact Information

Any question please contact the following email:
* 1035456235@qq.com
* YeaTmeRet@gmail.com
  
