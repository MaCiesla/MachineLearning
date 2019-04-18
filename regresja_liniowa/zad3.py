# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:17:53 2019

@author: Maciej Ciesla
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model as linm
from sklearn import preprocessing as pre




def F(x,theta):
    return theta*x
# Import danych
# Reggression models
# http://scikit-learn.org/stable/modules/linear_model.html
 
# Load the diabetes dataset
boston = datasets.load_boston()
# print description
print(boston.DESCR)
# get the data
boston_X = boston.data
boston_Y = boston.target


# Normalizacja/Standaryzacja

scaler = pre.StandardScaler()
boston_X = scaler.fit_transform(boston_X)
boston_Y = (boston_Y - np.mean(boston_Y))/np.std(boston_Y)

#zad1 Podział na zbiór treningowy i testowy (70-30%)
train_X = boston_X[:354,:]
train_Y = boston_Y[:354]
test_X = boston_X[:152,:]
test_Y = boston_Y[:152]

#zad2
## Stworzenie obiektu 
regr = linm.LinearRegression()
 
## Uczenie modelu przy pomocy bazy treningowej
regr.fit(train_X, train_Y)
## Przewidywanie wartości dla danych testowych
Y_predicted = regr.predict(test_X)
 
## Wyświetlenie parametrów prostej
print('Coefficients: \n', regr.coef_)
 
##  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy
error = np.mean((regr.predict(test_X) - test_Y) ** 2)
print("Residual sum of squares: {}".format(error))

#zad3 wizualizacja prostych regresji
name = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

for i in range(0,13):
    xreg = np.matrix(np.linspace(np.min(test_X[:,i]), np.max(test_X[:,i]), num=100))
    xreg_temp = np.append(np.ones((1,100)), xreg, axis=0)
    yreg = F(xreg_temp, [regr.intercept_, regr.coef_[i]])
    
    plt.figure()
    plt.plot(xreg.T, yreg.T,color='r', linewidth=3)
    plt.scatter(test_X[:,i], test_Y)
    plt.xlabel(name[i])
    plt.ylabel('MEDV')
    plt.show()



#zad4 Porównanie modeli regresji
reg_LinReg =linm.LinearRegression()
reg_Ridge = linm.Ridge(alpha = .5)
reg_Lasso = linm.Lasso(alpha = 5.1)
reg_ElNet =linm.ElasticNet(alpha = .5, l1_ratio=0.5)

reg_LinReg.fit(train_X, train_Y)
reg_Ridge.fit(train_X, train_Y)
reg_Lasso.fit(train_X, train_Y)
reg_ElNet.fit(train_X, train_Y)

Y_predicted_LinReg = reg_LinReg.predict(test_X)
Y_predicted_Ridge = reg_Ridge.predict(test_X)
Y_predicted_Lasso = reg_Lasso.predict(test_X)
Y_predicted_ElNet = reg_ElNet.predict(test_X)

error = np.mean((Y_predicted_LinReg - test_Y) ** 2)
print("Residual sum of squares for LinReg: {}".format(error))
error = np.mean((Y_predicted_Ridge - test_Y) ** 2)
print("Residual sum of squares for Ridge: {}".format(error))
error = np.mean((Y_predicted_Lasso - test_Y) ** 2)
print("Residual sum of squares for Lasso: {}".format(error))
error = np.mean((Y_predicted_ElNet - test_Y) ** 2)
print("Residual sum of squares for ElNet: {}".format(error))



