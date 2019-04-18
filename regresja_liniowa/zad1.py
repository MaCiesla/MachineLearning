# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:24:47 2019

@author: Maciej Ciesla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def F(x,theta):
    return theta*x


#zad3 zaimplementowana funkcja computeCost
def computeCost(X, y, theta):
    m = X.shape[1]
    suma = np.sum(np.power((F(X,theta)-y),2))
    J = 0.5*suma/m
    return J


#zad7 zaimplementowana metoda gradientu prostego
def gradient_prosty(X, y, theta, alpha, it):
    m = X.shape[1]
    cost = computeCost(X,y,theta)
    for i in range(it):
        theta = theta - alpha*np.sum((F(X,theta)-y)*np.transpose(X),axis=0)/m
        cost = np.append(cost, computeCost(X,y,theta))
    return theta, cost


alpha = 0.01
it = 1000


path = os.getcwd() + '/dane1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

#zad1
print(data.describe())
print(data.head(10))

#zad2 wykres danych
plt.figure()
plt.scatter(data['Population'],data['Profit'])
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

#zad5 podzielenie danych
X = data['Population']
y = data['Profit']



X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0], dtype='f'))

#zad4 dodadanie kolumny jedynek
X = np.append(np.ones((1,X.shape[1])), X, axis=0)

#zad6 wyliczenie kosztu dla theta = [0,0]
J = computeCost(X,y,theta)
print("koszt dla theta = [0,0] to",J)


#zad8 wylicznone optymalne parametry
grad = gradient_prosty(X,y,theta,alpha,it)
print("wyliczone optymalne parametry to ", grad[0])

#zad9 wartoć funkcji kosztu dla optymalnych parametrów
print("wartoć funkcji kosztu dla optymalnych parametrów to", grad[1][len(grad[1])-1])

#zad10 wykres regresji liniowej i danych
xreg = np.matrix(np.linspace(4.0, 23.0, num=100))
xreg_temp = np.append(np.ones((1,100)), xreg, axis=0)
yreg = F(xreg_temp, grad[0])

plt.figure()
plt.plot(xreg.T, yreg.T,color='r', linewidth=3)
plt.scatter(data['Population'],data['Profit'])
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

#zad11 wykres zależnoci funkcji kosztu od iteracji
plt.figure()
plt.plot(grad[1])
plt.xlabel('Iteracje')
plt.ylabel('Koszt')
plt.show()
