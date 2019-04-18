# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:15:49 2019

@author: student191
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model as linm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

seed = 20
dataset = pd.read_csv('boston.csv.pdf')
 
X = dataset.drop('MEDV', axis=1)
y = dataset['MEDV']


#standaryzacja danych
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
yn = (y-np.mean(y))/np.std(y)
#podziałz danych
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3, random_state = seed)

#regresja liniowa
regr = linm.LinearRegression()
regr.fit(features_train, labels_train)
predict = regr.predict(features_test)


output = regr.score(features_train, labels_train)
print("skutecznosc",output)

output = regr.score(features_test, labels_test)
print("skutecznosc testowa",output)


#wielomiany

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linm.LinearRegression())
]
poly = Pipeline(steps)
poly.fit(features_train, labels_train)


output = poly.score(features_train, labels_train)
print("skutecznosc z wiel",output)

output = poly.score(features_test, labels_test)
print("skutecznosc testowa z wiel",output)

#Ridge
clf = Ridge(alpha=10)
clf.fit(features_train, labels_train) 


output = clf.score(features_train, labels_train)
print("skutecznosc z Ridge",output)

output = clf.score(features_test, labels_test)
print("skutecznosc testowa z Ridge",output)


al = np.logspace(-3,4,100)
out = np.zeros((100,1))
maxal = -1
maxout = -1
for i in range (al.size):
    clf = Ridge(alpha=al[i])
    clf.fit(features_train, labels_train) 
    output = clf.score(features_test, labels_test)
    out[i] = output
    if maxout == -1:
        maxout =  output
        maxal = al[i]
    else:
        if output > maxout:
            maxout = output
            maxal = al[i]
        

plt.figure()
plt. plot(al,out)
plt.xlabel("alpha")
plt.ylabel("skutecznosc")
plt.show()
print("maksymalna skutecznosc: ",maxout)
print("optymalna alpha: ", maxal)


#Lasso
maxi = 0
a = 0
t = np.linspace(0,1,100)
for i in range (t.size):
    clf = linm.Lasso(alpha=t[i])
    clf.fit(features_train, labels_train) 
    
    output = clf.score(features_test, labels_test)
    if output > maxi:
        maxi=output
        a = t[i]

clf = linm.Lasso(alpha=a)
clf.fit(features_train, labels_train) 
    
output = clf.score(features_train, labels_train)
print("skutecznosc z Lasso",output)

output = clf.score(features_test, labels_test)
print("skutecznosc testowa z Lasso",output)

print("optymalna alpha: ",a)

