# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:13:11 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Boosting decision trees for classification
data=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/bagging_boosting_stacking_RF/pima-indians-diabetes.data.csv')
data.columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
array=data.values
X=array[:,0:8]
Y=array[:,8]

#splitting data using K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)

#building decision tree classifier using adaboost
from sklearn.tree import DecisionTreeClassifier
cart=DecisionTreeClassifier(criterion='entropy')
from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier(base_estimator=cart,n_estimators=10,random_state=7)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#0.69
