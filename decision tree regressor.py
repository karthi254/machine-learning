# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:02:49 2023

@author: gunak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("diabetes.csv")

x = df.iloc[:,:-1].astype(int)
y = df.iloc[:,7].astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

dc = DecisionTreeRegressor()
dc.fit(x_train,y_train)

y_pred = dc.predict(x_test).astype(int)


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

from sklearn import tree
fig = plt.figure()
tree.plot_tree(dc)
plt.show()
fig.savefig("desition_tree R.png")


