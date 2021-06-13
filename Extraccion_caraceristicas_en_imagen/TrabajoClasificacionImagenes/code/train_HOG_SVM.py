#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:25:27 2020

@author: nico

Script: train_HOG_SVM.py
Descripción: Entrena un clasificador SVM que identifica imágenes con peatones
			e imágenes de fondo haciendo uso de descriptores HOG. El
			entrenamiento se lleva a cabo haciendo uso de una partición fija
			de entrenamiento y test.
Uso: train_HOG_SVM.py -d <Documento JSON de experimentación> [-s]


El script recibe un documento JSON de experimentación con la siguiente
estructura:
	
	- kernel_type -> Tipo de kernel de SVM. str con alguno de los siguientes
					valores: SVM_LINEAR, SVM_POLY, SVM_RBF y SVM_SIGMOID
	- Cvalue -> Valor de la constante de regularización: float mayor
				que 0.
	- degree -> Grado del polinomio para el kernel polinómico: int mayor que 0
	- gamma -> Valor de gamma para los kernels rbf, sigmoide y polinómico:
				float
	- coef0 -> Término independiente para los kernel sigmoidal y polinómico:
				float

Por su parte, si se especifica el parámetro -s, todo modelo generado se
almacenará.

"""

# Librerías incluídas
import sys
import time
import json
import argparse
from eci import (load_image_class_dataset, compute_HOG, train_svm, test_svm,
					check_experiment_document, extract_experiments_parameters)

PATH_POSITIVE_TRAIN = 'data/train/pedestrians/'
PATH_NEGATIVE_TRAIN = 'data/train/background/'
PATH_POSITIVE_TEST = 'data/test/pedestrians/'
PATH_NEGATIVE_TEST = 'data/test/background/'


#### Cuerpo del script ####

## Leer argumentos pasados al script
parser = argparse.ArgumentParser(description='Entrena un clasificador SVM que'\
								 ' identifica imágenes con peatones e imágenes de'\
								 ' fondo haciendo uso de descriptores HOG')
parser.add_argument('-d', '--document', help='Fichero JSON con los parámetros'\
					' del entrenamiento a considerar', type=str)
parser.add_argument('-s', '--save_model', help='Establecer para almacenar los'\
					' modelos generados durante el entrenamiento',
					action='store_true', default=False)

args = parser.parse_args()

exp_filename = args.document
store_models = args.save_model

# Leer fichero de experimentación
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

# Determinar el nombre del fichero destino y del modelo destino
dot_pos = exp_filename.rfind('.')
if dot_pos != -1:
	results_filename = exp_filename[:dot_pos] + '_experimentos.json'
	model_base_filename = exp_filename[:dot_pos]
else:
	results_filename = exp_filename + '_experimentos.json'
	model_base_filename = exp_filename[:]


## Carga de imágenes y extracción de descriptores HOG

# Cargar las imágenes de training
X_train, y_train = load_image_class_dataset({1: PATH_POSITIVE_TRAIN,
                                              0: PATH_NEGATIVE_TRAIN},
                                                compute_HOG)
    
# Cargar las imágenes de test
X_test, y_test = load_image_class_dataset({1: PATH_POSITIVE_TEST,
                                            0: PATH_NEGATIVE_TEST},
                                                compute_HOG)

## Extracción de todas las combinaciones de parámetros
svm_params = extract_experiments_parameters(exp_data)

results = []

for p in svm_params:

	print('Entrenando con parámetros: {}'.format(p))

	# Entrenamiento de SVM
	t_start = time.time()
	svm = train_svm(X_train, y_train, svm_params=p)
	t_end = time.time()
	
	# Evaluar modelo entrenado
	meausures = test_svm(svm, X_test, y_test)
	
	# Almacenar el modelo
	if store_models:
		svm.save(model_base_filename + '-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_modelo.dat')
	
	# Almacenar toda la información de parámetros y resultados
	p['resultados'] = meausures
	results.append(p)

# Almacenar los datos en un JSON
with open(results_filename, 'w') as f:
	json.dump(results, f, indent=4)

# Almacenar los resultados
#save_results('modelo1.dat', meausures)
