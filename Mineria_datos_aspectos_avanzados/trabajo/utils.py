# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:29:32 2020

@author: nico
"""
import os
import os.path
from tensorflow.keras import backend as K
import numpy as np
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def extract_experiments_parameters(experiment: dict):

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
		
		index_params['batch_size'] = 0

	if 'lr' in experiment:
		if not isinstance(experiment['lr'], list):
			v_params['lr'] = [experiment['lr']]
		else:
			v_params['lr'] = experiment['lr']
		
		index_params['lr'] = 0

	if 'conv2d_1_filters' in experiment:

		if not isinstance(experiment['conv2d_1_filters'], list):
			v_params['conv2d_1_filters'] = [experiment['conv2d_1_filters']]
		else:
			v_params['conv2d_1_filters'] = experiment['conv2d_1_filters']
		
		index_params['conv2d_1_filters'] = 0

	if 'conv2d_2_filters' in experiment:

		if not isinstance(experiment['conv2d_2_filters'], list):
			v_params['conv2d_2_filters'] = [experiment['conv2d_2_filters']]
		else:
			v_params['conv2d_2_filters'] = experiment['conv2d_2_filters']
		
		index_params['conv2d_2_filters'] = 0

	if 'conv2d_3_filters' in experiment:

		if not isinstance(experiment['conv2d_3_filters'], list):
			v_params['conv2d_3_filters'] = [experiment['conv2d_3_filters']]
		else:
			v_params['conv2d_3_filters'] = experiment['conv2d_3_filters']
		
		index_params['conv2d_3_filters'] = 0

	if 'conv2d_4_filters' in experiment:

		if not isinstance(experiment['conv2d_4_filters'], list):
			v_params['conv2d_4_filters'] = [experiment['conv2d_4_filters']]
		else:
			v_params['conv2d_4_filters'] = experiment['conv2d_4_filters']
		
		index_params['conv2d_4_filters'] = 0

	if 'conv2d' in experiment:

		if not isinstance(experiment['conv2d'], list):
			v_params['conv2d'] = [experiment['conv2d']]
		else:
			v_params['conv2d'] = experiment['conv2d']
		
		index_params['conv2d'] = 0

	if 'dense' in experiment:

		if not isinstance(experiment['dense'], list):
			v_params['dense'] = [experiment['dense']]
		else:
			v_params['dense'] = experiment['dense']
		
		index_params['dense'] = 0

	if 'modelo' in experiment:

		if not isinstance(experiment['modelo'], list):
			v_params['modelo'] = [experiment['modelo']]
		else:
			v_params['modelo'] = experiment['modelo']
		
		index_params['modelo'] = 0

	# Construir diccionarios univalores por cada combinación de parámetros
	keys = list(v_params.keys())
	
	while index_params[keys[0]] < len(v_params[keys[0]]):

		aux = {}

		# Insertar epochs
		if epochs is not None:
			aux['epochs'] = epochs

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

def mean_f1_score(y_true, y_pred, return_conf_matrix=False):
	
	# Calcular matriz de confusión
	conf_matrix = confusion_matrix(y_true, y_pred)
	n_classes = conf_matrix.shape[0]

	scores = np.zeros((n_classes,))

	for c in range(n_classes):
		
		# Calcular matriz de confusión binaria para cada clase
		#bin_conf_matrix = K.zeros((2,2), dtype='int32')
		
		TP = conf_matrix[c, c]
		FP = conf_matrix[:, c].sum() - conf_matrix[c, c]
		FN = conf_matrix[c, :].sum() - conf_matrix[c, c]
	
		# Calcular precision y recall
		precision = TP / (TP+FP)
		recall = TP / (TP+FN)
		
		# Calcular f1_score
		scores[c] = 2*precision*recall/(precision+recall)
	
	return scores.mean() if not return_conf_matrix else scores.mean(), conf_matrix

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

def save_prediction(ids: list, labels: list, filename: str):

	# Crear archivo en el que se almacenarán los datos
	f = open(filename, 'w')

	# Escribir cabecera
	f.write('id.jpg,label\n')

	# Escribir cada una de las prediciones
	for i in range(len(ids)):
		f.write('{},{}\n'.format(ids[i], labels[i]))

	f.close()

class GenetarorCombiner:

	"""
		Combinador de generadores de imágenes

		Parámetros:
			- gens: Lista de generadores de imágenes
			- ratio_reps: Lista de repeticiones en el número de veces que
						se extrae un batch de cada generador por iteración.
						Si no se especifica, se asumirá 1 para cada generador
	"""
	
	def __init__(self, gens: tuple or list, ratio_reps=None):

		# Atributos privados
		self.__gens = gens
		self.__ratio_reps = ratio_reps

		if not self.__ratio_reps:
			self.__ratio_reps = (1,) * len(self.__gens)

		# Definir turnos de extracción de cada generador
		self.__turns = []
		for i in range(len(self.__gens)):
			self.__turns.extend( [self.__gens[i]]*self.__ratio_reps[i] )

	def generate(self):
		# Obtener el generador y extraer los batchs
		while True:
			for g in self.__turns:
				yield(next(g))

	def __len__(self):
		return sum(len(g) for g in self.__turns)
