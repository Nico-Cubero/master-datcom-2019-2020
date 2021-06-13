#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:25:27 2020

@author: nico

Script: train_HOG_SVM_kfold.py
Descripción: Entrena clasificadores SVM que identifica imágenes con peatones
			e imágenes de fondo haciendo uso de descriptores HOG y evalúa
			su accuracy medio considerando un particionamiento kfold
			estratificado. Este script no guarda ningún modelo generado
Uso: train_HOG_SVM_kfold.py <Documento JSON de experimentación>


El script recibe un documento JSON de experimentación con la siguiente
estructura:

	- kfold_splits -> Número de particiones a considerar en kfold. Debe de
					ser un valor entero mayor que 0.
	- kernel_type -> Tipo de kernel de SVM. str con alguno de los siguientes
					valores: SVM_LINEAR, SVM_POLY, SVM_RBF y SVM_SIGMOID
	- Cvalue -> Valor de la constante de regularización: float mayor
				que 0.<<
	- degree -> Grado del polinomio para el kernel polinómico: int mayor que 0
	- gamma -> Valor de gamma para los kernels rbf, sigmoide y polinómico:
				float
	- coef0 -> Término independiente para los kernel sigmoidal y polinómico:
				float

"""

# Librerías incluídas
#import cv2 as cv
#import numpy as np
import sys
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from eci import (load_image_class_dataset, compute_HOG, train_svm,
				 check_experiment_document, extract_experiments_parameters)

PATH_POSITIVE_TRAIN = 'data/train/pedestrians/'
PATH_NEGATIVE_TRAIN = 'data/train/background/'
PATH_POSITIVE_TEST = 'data/test/pedestrians/'
PATH_NEGATIVE_TEST = 'data/test/background/'

def train_SVM_kfold(X, y, partitions, train_f, svm_params=None, verbose=False):
	
	results = []
	times = []
	
	k = 1 # Contar particiones efectuadas
	for train_index, test_index in partitions:
		
		# Tomar la partición actual de train y test
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]
		
		if verbose:
			print('Entrenando con partición {}'.format(k))
		
		# Entrenar con el conjunto de train
		t_start = time.time()
		svm = train_f(X_train, y_train, svm_params)
		times.append(time.time()-t_start)
		
		# Evaluar el modelo
		y_pred = svm.predict(X_test)[1].squeeze()
		score = np.mean(y_pred == y_test)

		if verbose:
			print('Evaluación realizada con accuraccy {}. '\
				  'Experimento realizado en {} s'.format(score, times[-1]))
		
		results.append(score)

		k += 1
	
	# Computar scores medios
	return {
			 'avg_score': sum(results)/len(results),
			 'min_score': min(results),
			 'max_score': max(results),
			 'avg_elapsed_time': sum(times)/len(times),
			 'min_elapsed_time': min(times),
			 'max_elapsed_time': max(times)
			 }

#### Cuerpo del script ####

# Leer fichero de experimentación
if len(sys.argv) != 2:
	print('Especificar fichero JSON con la experimentación:'\
   ' {} <fichero JSON experimentación>'.format(sys.argv[0]), file=sys.stderr)
	exit(-1)

exp_filename = sys.argv[1]

with open(exp_filename) as f:
	try:
		exp_data = json.load(f)
	except Exception as e:
		print('Se produjo un error con el fichero de experimentación'\
			' expecificado:\n',str(e), file=sys.stderr)
		exit(-1)

# Comprobar el formato del fichero de documentación
try:
	check_experiment_document(exp_data)
except Exception as e:
	print('El documento de experimentación no es correcto:', file=sys.stderr)
	print(str(e), file=sys.stderr)
	exit(-1)

# Determinar el nombre del fichero destino
dot_pos = exp_filename.rfind('.')
if dot_pos != -1:
	results_filename = exp_filename[:dot_pos] + '_experimentos.json'
else:
	results_filename = exp_filename + '_experimentos.json'

## Carga de imágenes y extracción de descriptores HOG

# Cargar las imágenes de training
X_train, y_train = load_image_class_dataset({1: PATH_POSITIVE_TRAIN,
											  0: PATH_NEGATIVE_TRAIN},
												compute_HOG)
	
# Cargar las imágenes de test
X_test, y_test = load_image_class_dataset({1: PATH_POSITIVE_TEST,
											0: PATH_NEGATIVE_TEST},
												compute_HOG)

## Particionamiento KFold estratificado del conjunto de train
kfold_splits = exp_data['kfold_splits'] if 'kfold_splits' in exp_data else 5
kfold = StratifiedShuffleSplit(n_splits=kfold_splits, random_state=11)
kf_split = list(kfold.split(X_train, y_train))


## Extracción de todas las combinaciones de parámetros
svm_params = extract_experiments_parameters(exp_data)

results = []

for p in svm_params:#np.linspace(0.01, 10, 10):#, np.logspace(-1,1,10)):
	#svm_params = {'Cvalue': C, 'kernel_type': SVM_LINEAR}#, 'gamma': gamma}
	print('Entrenando con parámetros:\n', p)
	result = train_SVM_kfold(X_train, y_train, kf_split, train_svm, p, verbose=True)
	
	# Almacenar los resultados junto a los parámetros
	p['resultados'] = result
	
	results.append(p)


# Almacenar los datos en un JSON
with open(results_filename, 'w') as f:
	json.dump(results, f, indent=4)
