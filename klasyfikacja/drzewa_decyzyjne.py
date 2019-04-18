# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:14:33 2019

@author: Maciej Ciesla
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
import graphviz


#wczytanie danych
iris = datasets.load_iris()


clf = DecisionTreeClassifier()
#trenowanie modelu
model = clf.fit(iris.data, iris.target)

print(clf.predict(iris.data[:-1, :]))
print(clf.predict_proba(iris.data[:-1, :]))

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)
#rysowanie grafu
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("iris.pdf")
















