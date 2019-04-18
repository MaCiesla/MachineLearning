# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:45:05 2019

@author: Maciej Ciesla
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


iris = datasets.load_iris()


#Podziel zbiór na uczący i testowy, test_size - procentowy udział
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)


#wykorzystanie biblioteki  sklearn.neighbors.KNeighborsClassifier
Kspace = np.linspace(1,50,50)
outputs = []
Kmax=0
maxout = -1
for K in Kspace:
    nbrs = KNeighborsClassifier(n_neighbors=np.int(K)).fit(features_train, labels_train)
    result = nbrs.predict(features_test)
    output = accuracy_score(labels_test, result)
    outputs.append(output)
    if maxout == -1:
        maxout =  output
        Kmax = K
    else:
        if output > maxout:
            maxout = output
            Kmax = K

plt.figure()
plt.plot(Kspace,outputs)
plt.xlabel('K')
plt.ylabel('skutecznosc')
plt.show

print("maksymalna skutecznosc to: ",maxout,"dla K = ",Kmax)


