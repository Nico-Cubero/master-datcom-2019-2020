# -*- coding: utf-8 -*-

## Importar librerías
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
													recall_score)

np.random.seed(7)

## Cargar datos de iris
iris = load_iris()
X, y = iris['data'], iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

## Entrenar con un modelo SVM
print('Entrenando con modelo SVM')
svm = SVC(kernel='rbf', C=2.2, gamma=0.21)
svm.fit(X_train, y_train)

## Evaluar y predecir el conjunto de test
print('Prediciendo un conjunto de test')
score = svm.score(X_test, y_test)
print('Se ha obtenido un Accuracy de {}'.format(score))

y_predict = svm.predict(X_test)

print('Matriz de confusión: Filas->Clase real, Columnas->Clase predicha:')
print(confusion_matrix(y_test, y_predict))
print('Precisión media asociada al clasificador: ', precision_score(y_test,
												y_predict, average='weighted'))
print('Recall medio asociada al clasificador: ', recall_score(y_test,
												y_predict, average='weighted'))
print('Puntuación F1 media asociada al clasificador: ', f1_score(y_test,
												y_predict, average='weighted'))
