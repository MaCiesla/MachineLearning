# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:46:04 2019

@author: Maciej Ciesla
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score
 
iris = datasets.load_iris()
X = iris.data[:, :2]  # analizujemy tylko dwa parametry
Y = iris.target
#zad1
regr = LogisticRegression(max_iter=150,solver='sag', C=1, multi_class='multinomial')
#zad2
regr.fit(X,Y)


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = regr.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
 
plt.show()

#zad3
print("Prawdopodobienstwo")
print(regr.predict_proba(X))

#zad4
predict = regr.predict(X)
output = accuracy_score(Y, predict)
print("Skutecznosc",output)


