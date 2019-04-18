# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:46:42 2019

@author: student191
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = os.getcwd() + '/breast_cancer_data.txt'
dataset = pd.read_csv(path, header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

dataset['Class'].replace(2, 0, inplace=True)
dataset['Class'].replace(4, 1, inplace=True)

#zad1
print(np.sum(dataset.isnull()))

#zad2
X = dataset[dataset.columns[1:-1]]
y = dataset['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

#zad3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#zad4
regr = LogisticRegression(penalty = 'l1', max_iter=150, C=1)
regr.fit(X_train,y_train)
predict = regr.predict(X_test)

output = accuracy_score(y_test,predict)
print("skutecznosc dla l1",output)


weight, params, outputs = [],[],[]
for c in np.linspace(-4,1,10):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train,y_train)
    weight.append(lr.coef_)
    params.append(1/10.**c)
    predict = lr.predict(X_test)
    output = accuracy_score(y_test,predict)
    outputs.append(output)
weight = np.array(weight)

plt.figure()

for i in range(9):
    plt.plot(params,weight[:,:,i])
plt.legend(['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'])
plt.ylabel('wagi')
plt.xlabel('lambda')
plt.xscale('log')
plt.show()

plt.figure()
plt.plot(params,outputs)
plt.ylabel('skutecznosc')
plt.xlabel('lambda')
plt.xscale('log')
plt.show()


