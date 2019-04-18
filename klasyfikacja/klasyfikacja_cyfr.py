# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:22:53 2019

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

def plot_mnist(images, titles, h, w, n_row=3, n_col=3):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.05)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

 
 
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

#zad2 Proszę wyświetlić liczbę cyfr oraz liczbę pikseli przypadającą na jeden obraz
tab = [0,0,0,0,0,0,0,0,0,0]
for i in range(y.shape[0]):
    tab[int(y[i])] += 1
for i in range(10):
    print("cyfry:",i,"jest",tab[i])
print("Liczba pikseli na jeden obraz:", X.shape[1])

#zad3 wywietlenie przykładowych cyfr
plot_mnist([X[0,:],X[500,:],X[1000,:],X[1500,:],X[2000,:],X[2500,:],X[3000,:],X[3500,:],X[4000,:],X[4500,:],X[4999,:]],"012345678",h,w)

#zad4 podzial zbioru na treningowy i testowy
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3)


clf = DecisionTreeClassifier(random_state=0, max_depth=10)
model = clf.fit(features_train, labels_train)
predict = clf.predict(features_test)


print("prawdopodobienskwo")
print(clf.predict_proba(X[:-1, :]))


output = accuracy_score(labels_test, predict)
print("skutecznosc",output)
F1 = f1_score(labels_test, predict,average=None)
for i in range(len(F1)):
    print("F1 od", i, "wynosi:", F1[i])
print("Macierz błędów:")
print(confusion_matrix(labels_test, predict))
print("Raport klasyfikacji:")
print(classification_report(labels_test, predict))






