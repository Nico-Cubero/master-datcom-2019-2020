#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:45:09 2020

@author: nico
"""
import os
import random
from types import FunctionType
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread, resize #, HOGDescriptor
from cv2.ml import SVM_LINEAR, SVM_POLY, SVM_RBF, SVM_SIGMOID

def load_image_class_dataset(data: dict, prepro_operation: dict=None,
								desc_operation=None, n_samples: dict=None,
								rep_samples: dict or int=None):

	"""Función para cargar las imágenes de un dataset contenidas en un
		directorio junto a su clase

		Recibe:
			data: Diccionario con la siguiente estructura
				clase - ruta de directorio

				clase: Etiqueta a asignar a las imágenes cargadas del
					directorio
				ruta del directorio: str con la ruta del directorio que
					contiene las imágenes.

			operation: Función que define una transformación a aplicar
				sobre las imágenes leídas. Debe de recibir una matriz
				numpy como entrada y devolver una matriz numpy

		Devuelve:
			- images: numpy array con las imágenes cargadas
			- classes: numpy array con las clases de las imágenes
	"""

	images = []
	classes = []

	for cl in data:

		images_cl = []

		filenames_list = os.listdir(data[cl])

		# Seleccionar submuestra si se ha especificado
		if n_samples and filenames_list:
			if isinstance(n_samples, dict) and cl in n_samples:
				filenames_list = random.sample(filenames_list, n_samples[cl])
			elif isinstance(n_samples, int):
				filenames_list = random.sample(filenames_list, n_samples)

		# Tomar muestras repetidas de la misma imagen (si se ha especificado esto)
		if isinstance(rep_samples, dict) and cl in rep_samples:
			n_imgs_samples = rep_samples[cl]
		elif isinstance(rep_samples, int):
			n_imgs_samples = rep_samples
		else:
			n_imgs_samples = 1

		# Leer las imágenes de cada clase
		for img_filename in filenames_list:

			filename = data[cl] + img_filename   # Nombre del fichero de imagen

			# Leer imagen, aplicar operación de transformación
			# y almacenarla con su clase
			img = imread(filename)

			for rep_sample in range(n_imgs_samples):

				img_sample = img

				# Preprocesar la imagen
				if prepro_operation:
					if isinstance(prepro_operation, dict) and  cl in prepro_operation:
						img_sample = prepro_operation[cl](img_sample)
					elif isinstance(prepro_operation, FunctionType):
						img_sample = prepro_operation(img_sample)

				if desc_operation:
					img_sample = desc_operation(img_sample)

				images_cl.append(img_sample)

		images.extend(images_cl)
		classes.extend([cl]*len(images_cl))

	# Convertir los datos en numpy array
	images = np.array(images)
	classes = np.array(classes)

	return images, classes

#__hog = HOGDescriptor()

def compute_HOG(image: np.array):

	"""Calcula los descriptores HOG de una imagen
		Recibe:
			image: Imagen a partir de la cual se calculan los descriptores
		Devuelve:
			Descriptores de la imagen
	"""

	# Calcular los descriptores de la imagen
	#hog = HOGDescriptor()
	hist = __hog.compute(image)

	return hist

def check_experiment_document(experiment: dict):

	"""Función para comprobar que un fichero JSON de experimentación,
		presenta todos los campos necesarios y en los formatos adecuados

		El fichero de documentación debe de presentar los siguientes campos:
		- kfold_splits -> Número de particiones a considerar en kfold. Debe de
							ser un valor entero mayor que 0
		- kernel_type -> Tipo de kernel de SVM. str con salguno de los siguientes
						valores: SVM_LINEAR, SVM_POLY, SVM_RBF y SVM_SIGMOID
		- Cvalue -> Valor de la constante de regularización: float mayor
					que 0.
		- degree -> Grado del polinomio para el kernel polinómico: int mayor que 0
		- gamma -> Valor de gamma para los kernels rbf, sigmoide y polinómico:
					float
		- coef0 -> Término independiente para los kernel sigmoidal y polinómico:
					float
	"""

	if not isinstance(experiment, dict):
		raise ValueError('El documento de experimentación no es un'\
				   ' documento válido')

	if 'kfold_splits' in experiment:

		if not isinstance(experiment['kfold_splits'], int):
			raise ValueError('"kfold_splits" debe ser entero')

		if experiment['kfold_splits'] <= 0:
			raise ValueError('"kfold_splits" debe de ser mayor que 0')

	if 'kernel_type' in experiment and not isinstance(experiment['kernel_type'],
												   str):
		raise ValueError('"kernel_type" debe de ser str')

	if 'Cvalue' in experiment:

		if isinstance(experiment['Cvalue'], (float, int)):
			if experiment['Cvalue'] <= 0:
				raise ValueError('"Cvalue" debe de ser mayor que 0')

		elif isinstance(experiment['Cvalue'], list):

			for k in experiment['Cvalue']:
				if not isinstance(k, (float, int)) or k <= 0:
					raise ValueError('Algún valor de "Cvalue" no es un valor'\
									  ' mayor que 0')
		else:
			raise ValueError('"Cvalue" debe de ser un valor real mayor que 0'\
								'o una lista de valores reales mayores que 0')

	if 'degree' in experiment:

		if isinstance(experiment['degree'], int):
			if experiment['degree'] <= 0:
				raise ValueError('"degree" debe de ser mayor que 0')

		elif isinstance(experiment['degree'], list):

			for k in experiment['degree']:
				if not isinstance(k, int) or k <= 0:
					raise ValueError('Algún valor de "degree" no es un valor'\
									  ' mayor que 0')
		else:
			raise ValueError('"degree" debe de ser un entero mayor que 0'\
								'o una lista de enteros mayores que 0')

	if 'gamma' in experiment:

		if isinstance(experiment['gamma'], list):
			for k in experiment['gamma']:
				if not isinstance(k, (float, int)):
					raise ValueError('Algún valor de "gamma" no es un valor real')

		elif not isinstance(experiment['gamma'], (float, int)):
			raise ValueError('"gamma" debe de ser un valor real')

	if 'coef0' in experiment:

		if isinstance(experiment['coef0'], list):
			for k in experiment['coef0']:
				if not isinstance(k, (float, int)):
					raise ValueError('Algún valor de "coef0" no es un valor real')

		elif not isinstance(experiment['coef0'], (float, int)):
			raise ValueError('"coef0" debe de ser un valor real')

def extract_experiments_parameters(experiment: dict):

	params = [] # Almacenar todas las combinaciones de parámetros

	v_params = {}
	index_params = {}

	kernel_type = None

	if 'kernel_type' in experiment:
		if experiment['kernel_type'] == 'SVM_LINEAR':
			kernel_type = SVM_LINEAR
		elif experiment['kernel_type'] == 'SVM_POLY':
			kernel_type = SVM_POLY
		elif experiment['kernel_type'] == 'SVM_RBF':
			kernel_type = SVM_RBF
		elif experiment['kernel_type'] == 'SVM_SIGMOID':
			kernel_type = SVM_SIGMOID

	if 'Cvalue' in experiment:
		if not isinstance(experiment['Cvalue'], list):
			v_params['Cvalue'] = [experiment['Cvalue']]
		else:
			v_params['Cvalue'] = experiment['Cvalue']

		index_params['Cvalue'] = 0

	if 'degree' in experiment:
		if not isinstance(experiment['degree'], list):
			v_params['degree'] = [experiment['degree']]
		else:
			v_params['degree'] = experiment['degree']

		index_params['degree'] = 0

	if 'gamma' in experiment:
		if not isinstance(experiment['gamma'], list):
			v_params['gamma'] = [experiment['gamma']]
		else:
			v_params['gamma'] = experiment['gamma']

		index_params['gamma'] = 0

	if 'coef0' in experiment:
		if not isinstance(experiment['coef0'], list):
			v_params['coef0'] = [experiment['coef0']]
		else:
			v_params['coef0'] = experiment['coef0']

		index_params['coef0'] = 0

	# Construir diccionarios univalores por cada combinación de parámetros
	keys = list(v_params.keys())

	while index_params[keys[0]] < len(v_params[keys[0]]):

		aux = {}

		# Insertar kernel_type
		if kernel_type is not None:
			aux['kernel_type'] = kernel_type

		# Introducir una combinación de parámetros
		for k in keys:
			aux[k]=v_params[k][index_params[k]]

		params.append(aux)

		# Incrementar índice del último parámetro
		index_params[keys[-1]] += 1

		# Completar contadores
		for i in range(len(keys)-1,0,-1):

			if index_params[keys[i]] == len(v_params[keys[i]]):
				# Resetear el contador del parámetro actual
				index_params[keys[i]] = 0

				# Avanzar al siguiente valor del anterior parámetro
				index_params[keys[i-1]] += 1

	return params

def extract_experiments_cnn_parameters(experiment: dict):

	params = [] # Almacenar todas las combinaciones de parámetros

	v_params = {}
	index_params = {}

	epochs = None

	if 'epochs' in experiment:
		epochs = experiment['epochs']

	if 'batch_size' in experiment:
		if not isinstance(experiment['batch_size'], list):
			v_params['batch_size'] = [experiment['batch_size']]
		else:
			v_params['batch_size'] = experiment['batch_size']

	else:
		v_params['batch_size'] = [None]

	index_params['batch_size'] = 0

	if 'lr' in experiment:
		if not isinstance(experiment['lr'], list):
			v_params['lr'] = [experiment['lr']]
		else:
			v_params['lr'] = experiment['lr']

		index_params['lr'] = 0

	# Construir diccionarios univalores por cada combinación de parámetros
	keys = list(v_params.keys())

	while index_params[keys[0]] < len(v_params[keys[0]]):

		aux = {}

		# Insertar epochs
		if epochs is not None:
			aux['epochs'] = epochs

		# Insertar arquitectura
		if 'arquitectura' in experiment:
			aux['arquitectura'] = experiment['arquitectura']
		else:
			aux['arquitectura'] = None

		if 'load_weights' in experiment:
			aux['load_weights'] = bool(experiment['load_weights'])
		else:
			aux['load_weights'] = True

		# Introducir una combinación de parámetros
		for k in keys:
			aux[k]=v_params[k][index_params[k]]

		params.append(aux)

		# Incrementar índice del último parámetro
		index_params[keys[-1]] += 1

		# Completar contadores
		for i in range(len(keys)-1,0,-1):

			if index_params[keys[i]] == len(v_params[keys[i]]):
				# Resetear el contador del parámetro actual
				index_params[keys[i]] = 0

				# Avanzar al siguiente valor del anterior parámetro
				index_params[keys[i-1]] += 1

	return params


def no_pederestian_img_prepro(img: np.array):

	# Reescalar las imágenes a un tamaño de 512x512
	img = resize(img, (512,512))

	# Seleccionar un trozo de 128x64
	i = random.randint(0, img.shape[0]-128)
	j = random.randint(0, img.shape[1]-64)

	return img[i:i+128, j:j+64]

def confusion_matrix(y_true, y_pred):

	# Situar vectores de etiqueta en una dimensión
	y_true = y_true.squeeze().astype('int32')
	y_pred = y_pred.squeeze().astype('int32')

	n_classes = y_true.max()+1

	# Matriz en la que se almacena el resultado
	conf_matrix = np.zeros((n_classes, n_classes), dtype='int64')

	# Anotar la comparación en la matriz
	for i in range(y_true.size):
		conf_matrix[y_true[i], y_pred[i]] += 1

	return conf_matrix

def plot_results(results: dict, metric: str, filename: str):

	plt.figure()

	# Imprimir cada gráfica
	for r in results:
		plt.plot(results[r])

	# Mostrar leyenda
	plt.title(metric)
	plt.xlabel('epoch')
	plt.ylabel(metric)
	plt.legend(results.keys(), loc='best')#, loc='lower right')

	plt.savefig(filename)

	plt.close() # Cerrar la figura una vez terminado
