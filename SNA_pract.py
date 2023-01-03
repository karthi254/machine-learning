# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:10:32 2022

@author: gunak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Social_Network_Ads.csv")
x = df.iloc[:,[1,2,3]].values
y = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
Labe = LabelEncoder()
x[:,0] = Labe.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression #for classification
Labe = LogisticRegression()
Labe.fit(x_train,y_train)

y_pred = Labe.predict(x_test)

from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test,y_pred)

Labe.score(x_train,y_train)
Labe.score(x_test,y_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(x_train,y_train)

knn = classifier.predict(x_test)
classifier.score(x_train,y_train)
classifier.score(x_test,y_test)
#accuracy1 = confusion_matrix(y_test,knn)
#knn.score()























