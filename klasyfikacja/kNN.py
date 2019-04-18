c# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:37:28 2019

@author: student191
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing as pre


K = 10
iris = datasets.load_iris()


#Podziel zbiór na uczący i testowy, test_size - procentowy udział (przykład 50 % uczący i testowy)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.5)




dist = np.zeros((features_train.shape[0]))
lab_check = [0,0,0]
result = np.zeros((labels_test.shape[0]))
 

for i in range (features_test.shape[0]):
    for j in range(features_train.shape[0]):          
        dist[j] = distance.euclidean(features_test[i,:],features_train[j,:])
    index = np.argsort(dist)
    indx = index[:K]
    lab_check = [0,0,0]
    for r in range(K):
        lab_check[labels_train[indx[r]]] += 1
    result[i] = np.argmax(lab_check)


# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, result)
print("własna metoda kNN:", output)

#wykorzystanie biblioteki  sklearn.neighbors.KNeighborsClassifier

nbrs = KNeighborsClassifier(n_neighbors=K).fit(features_train, labels_train)
result = nbrs.predict(features_test)
output = accuracy_score(labels_test, result)
print("metoda kNN z biblioteki", output)









