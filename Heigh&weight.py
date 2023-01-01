# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:52:49 2022

@author: gunak
"""
#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv("height_weight.csv")

#Split data into dependent and Independent variable
x = df.iloc[:,0:-1].values
y = df.iloc[:,1].values

#missing values detection
df.isnull().sum()
# fixing missing value
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy = "mean")
x = imp.fit_transform(x)

# check if the issue is fixed
np.isnan(x).sum()
#split dataset as train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state =0)

#fit the model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train,y_train)
y_pred = regress.predict(x_test)

plt.title("height and weight correlation on traning data")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regress.predict(x_train), color = "blue")
plt.show()

plt.title("height and weight correlation on traning data")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_test, regress.predict(x_test), color = "blue")
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import r2_score
accuracy = r2_score(y_test,y_pred)




my_ht = [[185]]
my_pred_wh = regress.predict(my_ht)




