# -*- coding: utf-8 -*-
"""

@author: Maciej Ciesla
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
 
 
# wczytywanie danych
dane = loadmat('baza_mnist.mat')
 
#Zad 1. Podziel dane na parametry X oraz odpowiedź y:
 
X = dane['X']
y = dane['y']
 
## Standaryzacja
for i in range(X.shape[0]):
    X[i,:] = (X[i,:]-np.mean(X))/np.std(X[i,:])
# 
# Zamiana cyfry 10 -> 0 (błąd w zbiorze danych)    
y[np.where(y==10)]=0
 
# wysokość i szerokość obrazka z cyfrą 
h = 20
w = 20

# podzial zbioru na treningowy i testowy
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3)


depthspace = np.linspace(1,50,50)
outputs = []
depthmax = 0
maxout = -1
for depth in depthspace:
    clf = DecisionTreeClassifier(random_state=0, max_depth=depth)
    model = clf.fit(features_train, labels_train)
    predict = clf.predict(features_test)
    output = accuracy_score(labels_test, predict)
    outputs.append(output)
    if maxout == -1:
        maxout =  output
        depthmax = depth
    else:
        if output > maxout:
            maxout = output
            depthmax = depth

plt.figure()
plt.plot(depthspace,outputs)
plt.xlabel('glebokosc')
plt.ylabel('skutecznosc')
plt.show

print("maksymalna skutecznosc to: ",maxout,"dla glebokosci = ",depthmax)










