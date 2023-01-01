# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:40:00 2022

@author: gunak
"""
import pandas as pd
df = pd.read_csv("50_Startups.csv")
x = df.iloc[:,:-1].values
y = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_x = LabelEncoder()
x[:,3] = LabelEncoder_x.fit_transform(x[:,3]) #string to int
onehotencoder_x =  OneHotEncoder(categories = [3]) #create a index
x = onehotencoder_x.fit_transform(x).toarray()

x = x[:,1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_pred) #output y_test and y_pred








