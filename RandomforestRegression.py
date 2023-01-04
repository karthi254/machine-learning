# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:30:39 2023

@author: gunak
"""

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("diabetes.csv")
x = df.iloc[:,:-1].astype(int)
y = df.iloc[:,7].astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
RG = RandomForestRegressor(n_estimators=10)
RG.fit(x_train,y_train)

y_pred = RG.predict(x_test).astype(int)

#from sklearn.preprocessing import StandardScaler
#std = StandardScaler()
#x_train = std.fit_transform(x_train)
#x_test = std.transform(x_test)


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))



from sklearn import tree
figure = plt.figure()
tree.plot_tree(RG.estimators_[5])
plt.show()
tree.plot_tree(RG.estimators_[0])
plt.show()











