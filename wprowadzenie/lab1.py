# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
# load the famous iris data
irisRaw = load_iris()
 
#Konwersja danych do pakietu Panda
# read iris.data into a pandas DataFrame (df), including column names
iris = pd.DataFrame(data= np.c_[irisRaw['data'], irisRaw['target']],
                     columns= irisRaw['feature_names'] + ['target'])

#zad1
print(iris)



#zad2
print("ilosc wierszy",iris.shape[0])
print("ilosc kolumn",iris.shape[1])


#zad3
print("podstawowe informacje dla poszczegolnych kolumn")
print(iris.describe())

print(iris.groupby('sepal length (cm)').mean())


#zad4
print("5 pierwszych wersji")
print(iris.head(5))


#zad5
print(iris.dropna(axis=1))
print("rozmiar zwroconej macierzy jest rowny rozmiarowi zbioru wiec nie ma danych brakujacych")

#zad6
print("dane posortowane rosnaco wedlug 2 kolumny")
sort = iris.sort_index(by=iris.columns[1])
print(sort)


#zad7
print("minimalna dlugosc platka = ",iris['petal length (cm)'].min())
print("indeks = ",iris['petal length (cm)'].idxmin())
print("maksymalna dlugosc platka = ",iris['petal length (cm)'].max())
print("indeks = ",iris['petal length (cm)'].idxmax())

#zad8
print("odchlenie standardowe")
print(iris.std())


#zad9 nie dziala jeszcze
part = iris['sepal length (cm)'] > iris['sepal length (cm)'].mean()
print(part)









