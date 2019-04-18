# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:39:54 2019

@author: Maciej Ciesla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre
from mpl_toolkits.mplot3d import Axes3D
import os

        

def F(x,theta):
    return theta*x

#zad3 zaimplementowana funkcja computeCost
def computeCost(X, y, theta):
    m = X.shape[1]
    suma = np.sum(np.power((F(X,theta)-y.T),2))
    J = 0.5*suma/m
    return J


#zad7 zaimplementowana metoda gradientu prostego
def gradient_prosty(X, y, theta, alpha, it):
    m = X.shape[1]
    cost = computeCost(X,y,theta)
    for i in range(it):
        theta = theta - alpha*np.sum((F(X,theta)-y.T)*np.transpose(X),axis=0)/m
        cost = np.append(cost, computeCost(X,y,theta))
    return theta, cost


alpha = 0.01
it = 1000


path = os.getcwd() + '/dane2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data.describe())
print(data.head())

#zad1 standaryzacja danych
print("Dane po standaryzacji")
scaler = pre.StandardScaler()
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, columns=['Size', 'Bedrooms', 'Price'])
print(data.describe())
print(data.head())

#zad2 wykres danych
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Size'], data['Bedrooms'], data['Price'])
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
plt.show()


#zad5 podzielenie danych
X = data[['Size','Bedrooms']]
y = data['Price']


X = np.transpose(np.matrix(X.values))
y = np.transpose(np.matrix(y.values))
theta = np.matrix(np.array([0,0,0], dtype='f'))


#zad4 dodadanie kolumny jedynek
X = np.append(np.ones((1,X.shape[1])), X, axis=0)

#zad6 wyliczenie kosztu dla theta = [0,0,0]
J = computeCost(X,y,theta)
print("koszt dla theta = [0,0,0] to",J)


#zad8 wylicznone optymalne parametry
grad = gradient_prosty(X,y,theta,alpha,it)
print("wyliczone optymalne parametry to ", grad[0])


#zad9 wartoć funkcji kosztu dla optymalnych parametrów
print("wartoć funkcji kosztu dla optymalnych parametrów to", grad[1][len(grad[1])-1])


#zad10 wykres regresji liniowej i danych
t = grad[0]

xreg = np.matrix(np.linspace(np.min(X[1,:]), np.max(X[1,:]), num=100))
xreg_temp = np.append(np.ones((1,100)), xreg, axis=0)
yreg = F(xreg_temp, [t[0,0], t[0,1]])
    
plt.figure()
plt.plot(xreg.T, yreg.T,color='r', linewidth=3)
plt.scatter(data['Size'], data['Price'])
plt.xlabel("Size")
plt.ylabel('Price')
plt.show()


xreg = np.matrix(np.linspace(np.min(X[2,:]), np.max(X[2,:]), num=100))
xreg_temp = np.append(np.ones((1,100)), xreg, axis=0)
yreg = F(xreg_temp, [t[0,0], t[0,2]])
    
plt.figure()
plt.plot(xreg.T, yreg.T,color='r', linewidth=3)
plt.scatter(data['Bedrooms'], data['Price'])
plt.xlabel("Bedrooms")
plt.ylabel('Price')
plt.show()


#zad11 wykres zależnoci funkcji kosztu od iteracji
plt.figure()
plt.plot(grad[1])
plt.xlabel('Iteracje')
plt.ylabel('Koszt')
plt.show()
