# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:41:34 2023

@author: gunak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv")

x = df.iloc[:,:-1].astype(int)
y = df.iloc[:,7].astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
regres = DecisionTreeClassifier(max_depth=2)
regres.fit(x_train,y_train)

y_pred = regres.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

from metrics import tree
fig = plt.figure()
tree.plot_tree(regres)
plt.show()
fig.savefig() 